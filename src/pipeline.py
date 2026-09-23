from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import uuid
import argparse
import gzip
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd
import numpy as np

from pulso_transmi import PulsoTransmiClient


ARTIFACT_DIR = Path("artifacts/baseline")
TRAIN_SCRIPT = Path("examples/03_supabase_linear_baseline.py")


def load_env(path: Path = Path(".env")) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key in values or key.startswith(("SUPABASE_", "PULSO_"))})
    return values


class SupabaseRestClient:
    def __init__(self, url: str, key: str) -> None:
        self.base_url = url.rstrip("/") + "/rest/v1"
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Prefer": "return=representation",
        }

    def request(self, method: str, table: str, payload: Any = None, *, params: dict[str, str] | None = None) -> list[dict[str, Any]]:
        query = f"?{urlencode(params)}" if params else ""
        request = Request(
            f"{self.base_url}/{table}{query}",
            data=None if payload is None else json.dumps(payload, default=str).encode(),
            headers=self.headers,
            method=method,
        )
        try:
            with urlopen(request) as response:
                result = json.loads(response.read())
        except Exception as exc:
            raise RuntimeError(f"Supabase {method} {table} failed: {exc}") from exc
        if not isinstance(result, list):
            raise RuntimeError(f"Supabase returned an unexpected payload for {table}")
        return result

    def latest_data_cut_id(self) -> int:
        rows = self.request("GET", "data_cut", params={"select": "id", "order": "queried_at.desc", "limit": "1"})
        if not rows:
            raise RuntimeError("No data_cut exists in Supabase")
        return int(rows[0]["id"])

    def find_model(self, version: str) -> dict[str, Any] | None:
        rows = self.request("GET", "model", params={"select": "id,version", "version": f"eq.{version}", "limit": "1"})
        return rows[0] if rows else None

    def active_model(self) -> dict[str, Any] | None:
        rows = self.request(
            "GET",
            "model",
            params={"select": "id,version,artifact_base64,artifact_sha256", "status": "eq.active", "order": "trained_at.desc", "limit": "1"},
        )
        return rows[0] if rows else None

    def latest_training_metrics(self, model_id: int) -> dict[str, Any]:
        rows = self.request(
            "GET",
            "training_run",
            params={"select": "metrics", "model_id": f"eq.{model_id}", "order": "started_at.desc", "limit": "1"},
        )
        return rows[0].get("metrics") or {} if rows else {}

    def insert(self, table: str, payload: dict[str, Any]) -> dict[str, Any]:
        rows = self.request("POST", table, payload)
        if not rows:
            raise RuntimeError(f"Supabase did not return the inserted {table}")
        return rows[0]

    def insert_many(self, table: str, payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not payload:
            return []
        return self.request("POST", table, payload)


def run_training() -> None:
    subprocess.run([os.environ.get("PYTHON", "python3"), str(TRAIN_SCRIPT)], check=True)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def forecast_open_cycle(api: PulsoTransmiClient, cycle: dict[str, Any], artifact: bytes) -> list[dict[str, Any]]:
    bundle = __import__("joblib").load(ARTIFACT_DIR / "model_and_metrics.joblib")
    stations = api.stations()
    context = api.context_dataframe(end=cycle["data_cutoff"])
    if context.empty:
        raise RuntimeError("No context available at the cycle cutoff")
    latest_context = context.sort_values("observed_at").iloc[-1]
    targets = [pd.Timestamp(target["target_at"]) for target in cycle["targets"]]
    rows = []
    for target in sorted(set(targets)):
        for station in stations.to_dict("records"):
            rows.append({
                "station_id": str(station["station_id"]),
                "corridor": station["corridor"],
                "latitude": station["latitude"],
                "longitude": station["longitude"],
                "observed_at": target,
                "event_intensity": latest_context["event_intensity"],
                "rain_forecast": latest_context["rain_forecast"],
                "rain_mm": latest_context["rain_mm"],
                "temperature_c": latest_context["temperature_c"],
                "temperature_forecast": latest_context["temperature_forecast"],
            })
    features = pd.DataFrame(rows)
    minutes = features["observed_at"].dt.hour * 60 + features["observed_at"].dt.minute
    features["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    features["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
    features["weekday_sin"] = np.sin(2 * np.pi * features["observed_at"].dt.dayofweek / 7)
    features["weekday_cos"] = np.cos(2 * np.pi * features["observed_at"].dt.dayofweek / 7)
    values = bundle["model"].predict(features[bundle["predictors"]])
    return [
        {"station_id": str(row["station_id"]).zfill(5), "target_at": pd.Timestamp(row["observed_at"]).isoformat(), "value": max(0.0, float(value))}
        for row, value in zip(features.to_dict("records"), values)
    ]


def restore_active_artifacts(db: SupabaseRestClient) -> None:
    active_model = db.active_model()
    if not active_model or not active_model.get("artifact_base64"):
        raise RuntimeError("No active model artifact is available in Supabase")
    stored_artifact = base64.b64decode(active_model["artifact_base64"])
    artifact = gzip.decompress(stored_artifact) if stored_artifact.startswith(b"\x1f\x8b") else stored_artifact
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    if artifact_sha256 != active_model.get("artifact_sha256"):
        raise RuntimeError("Active model artifact checksum mismatch")
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "model_and_metrics.joblib").write_bytes(artifact)
    metrics = {"test": db.latest_training_metrics(int(active_model["id"]))}
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))


def publish(env: dict[str, str], *, persist_evaluation: bool = True) -> dict[str, Any]:
    supabase_url = env.get("SUPABASE_URL")
    supabase_key = env.get("SUPABASE_SERVICE_ROLE_KEY") or env.get("SUPABASE_KEY")
    api_key = env.get("PULSO_API_KEY")
    api_url = env.get("PULSO_API_URL")
    if not supabase_url or not supabase_key or not api_key:
        raise RuntimeError("SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY and PULSO_API_KEY are required")

    metrics = json.loads((ARTIFACT_DIR / "metrics.json").read_text())
    predictions_path = ARTIFACT_DIR / "predictions.csv"
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    artifact = (ARTIFACT_DIR / "model_and_metrics.joblib").read_bytes()
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    version = f"linear-{artifact_sha256[:12]}"
    now = utc_now()
    db = SupabaseRestClient(supabase_url, supabase_key)
    data_cut_id = db.latest_data_cut_id()
    existing_model = db.find_model(version)
    if existing_model:
        model_id = int(existing_model["id"])
        run_row = {"id": "reused"}
        print(f"[inference] Reusing model version={version} id={model_id}")
    else:
        algorithm = metrics.get("selection", {}).get("selected_model", "unknown")
        compressed_artifact = gzip.compress(artifact, compresslevel=9)
        print(f"[inference] Artifact compressed: {len(artifact)} -> {len(compressed_artifact)} bytes")
        model_row = db.insert("model", {
            "version": version,
            "algorithm": f"sklearn.{algorithm}",
            "status": "active",
            "code_commit": os.getenv("GITHUB_SHA"),
            "trained_at": now.isoformat(),
            "artifact_base64": base64.b64encode(compressed_artifact).decode("ascii"),
            "artifact_sha256": artifact_sha256,
        })
        model_id = int(model_row["id"])
        run_row = db.insert("training_run", {
            "model_id": model_id,
            "data_cut_id": data_cut_id,
            "cutoff_at": now.isoformat(),
            "parameters": {"script": str(TRAIN_SCRIPT), "artifact_sha256": artifact_sha256},
            "metrics": metrics["test"],
            "started_at": now.isoformat(),
            "finished_at": now.isoformat(),
        })

    prediction_payload = [
        {
            "model_id": model_id,
            "station_id": str(row["station_id"]).zfill(5),
            "target_at": row["observed_at"],
            "generated_at": now.isoformat(),
            "horizon_periods": 1,
            "prediction": max(0, float(row["predicted_demand"])),
            "actual_value": float(row["actual_demand"]),
        }
        for row in predictions.to_dict("records")
    ]
    if persist_evaluation and not existing_model and not predictions.empty:
        prediction_rows: list[dict[str, Any]] = []
        for start in range(0, len(prediction_payload), 500):
            prediction_rows.extend(db.insert_many("prediction", prediction_payload[start : start + 500]))
        metric_payload = [
            {
                "prediction_id": int(row["id"]),
                "wape": metrics["test"]["wape"],
                "accuracy": metrics["test"]["accuracy"],
                "mae": metrics["test"]["mae"],
                "rmse": metrics["test"]["rmse"],
            }
            for row in prediction_rows
        ]
        for start in range(0, len(metric_payload), 500):
            db.insert_many("monitoring_metric", metric_payload[start : start + 500])
    print(f"[inference] Metrics: MAE={metrics['test']['mae']:.2f} RMSE={metrics['test']['rmse']:.2f} WAPE={metrics['test']['wape']:.4f}")

    with PulsoTransmiClient(base_url=api_url, api_key=api_key) as api:
        cycle = api.current_cycle()
        if cycle is None:
            print("[submission] No open cycle; publication skipped")
            return {
                "model_id": model_id,
                "training_run_id": run_row["id"],
                "submission": {"status": "skipped", "reason": "no_open_cycle"},
            }
        cycle_id = str(cycle.get("cycle_id") or cycle["id"])
        submission_predictions = forecast_open_cycle(api, cycle, artifact)
        payload = {
            "schema_version": "1.0",
            "cycle_id": cycle_id,
            "client_run_id": str(uuid.uuid4()),
            "data_cutoff": cycle["data_cutoff"],
            "model": {"version": version, "trained_at": now.isoformat(), "training_data_end": cycle["data_cutoff"], "git_commit": os.getenv("GITHUB_SHA")},
            "predictions": submission_predictions,
        }
        submission_key = f"{cycle_id}__{version}"
        submission = api.submit(payload, idempotency_key=submission_key)
        print(f"[submission] Sent cycle={cycle_id} model={version} predictions={len(submission_predictions)} status={submission.get('status')}")
    return {"model_id": model_id, "training_run_id": run_row["id"], "submission": submission}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    env = load_env()
    if args.publish_only:
        db = SupabaseRestClient(env["SUPABASE_URL"], env["SUPABASE_SERVICE_ROLE_KEY"])
        restore_active_artifacts(db)
        print("[inference] Restored active model artifact from Supabase")
        print(json.dumps(publish(env, persist_evaluation=False), indent=2, default=str))
    else:
        run_training()
        print(json.dumps(publish(env), indent=2, default=str))


if __name__ == "__main__":
    main()