from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import argparse
import gzip
import logging
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np

from pulso_transmi import PulsoTransmiClient
from src.accuracy import baseline_wape, recent_accuracy
from src.drift import paged_since
from src.lag_model import HISTORY_PERIODS, PERIOD, LagEnsembleForecaster
from src.mlflow_tracking import load_metadata
from src.supabase_resilience import SupabaseRestClient


ARTIFACT_DIR = Path("artifacts/baseline")
OUTBOX_DIR = Path("artifacts/outbox")
TRAIN_SCRIPT = Path("examples/03_supabase_linear_baseline.py")
REQUIRED_METRICS = ("mae", "rmse", "wape", "accuracy")
RETRAIN_ACCURACY_THRESHOLD = float(os.getenv("RETRAIN_ACCURACY_THRESHOLD", "79"))
# Minimum spacing between retrains, so a retrain that does not recover accuracy is not repeated every run.
RETRAIN_INTERVAL_HOURS = float(os.getenv("RETRAIN_INTERVAL_HOURS", "1"))
# Retrain also when production WAPE degrades this much versus the model's test WAPE (1.25 = 25 % worse).
RETRAIN_WAPE_RATIO = float(os.getenv("DRIFT_PERFORMANCE_RATIO", "1.25"))
LOG = logging.getLogger("pulso.pipeline")


class WaitingForOpenCycle(RuntimeError):
    """The API is healthy, but there is no cycle accepting submissions yet."""


class CycleAlreadySubmitted(RuntimeError):
    """The cycle already has an accepted submission; the API keeps only the first one."""


def is_idempotency_conflict(exc: Exception) -> bool:
    return getattr(exc, "status_code", None) == 409 and "idempotency_conflict" in str(exc)


def submitted_with_other_content(existing: dict[str, Any] | None, payload_sha256: str) -> bool:
    """True when the cycle was accepted earlier with a different payload (e.g. by a previous model).

    ``api_submission_id`` also counts: older runs overwrote confirmed rows with ``failed``
    after an idempotency conflict, but the API had accepted the first submission.
    """
    accepted = bool(existing and (existing.get("status") == "confirmed" or existing.get("api_submission_id")))
    return accepted and existing.get("payload_sha256") != payload_sha256


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


def run_training() -> None:
    environment = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (project_root, environment.get("PYTHONPATH")) if value
    )
    subprocess.run(
        [os.environ.get("PYTHON", "python3"), str(TRAIN_SCRIPT)],
        check=True,
        env=environment,
    )


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def usable_model(active_model: dict[str, Any] | None) -> bool:
    return bool(active_model and active_model.get("artifact_base64") and active_model.get("trained_at"))


def retrain_decision(
    active_model: dict[str, Any] | None,
    accuracy: float | None,
    now: datetime,
    threshold: float = RETRAIN_ACCURACY_THRESHOLD,
    interval_hours: float = RETRAIN_INTERVAL_HOURS,
    wape_ratio: float | None = None,
    ratio_threshold: float = RETRAIN_WAPE_RATIO,
) -> tuple[bool, str]:
    """Retrain when recent accuracy falls below the threshold, when production WAPE
    reaches ratio_threshold x the test WAPE, or when no model exists."""
    if not usable_model(active_model):
        return True, "no usable active model"
    causes = []
    if accuracy is not None and accuracy < threshold:
        causes.append(f"accuracy {accuracy:.2f}% < {threshold:g}%")
    if wape_ratio is not None and wape_ratio >= ratio_threshold:
        causes.append(f"WAPE ratio {wape_ratio:.2f} >= {ratio_threshold:g}")
    if not causes:
        if accuracy is None and wape_ratio is None:
            return False, "recent accuracy unavailable; keeping the active model"
        measured = [f"accuracy {accuracy:.2f}% >= {threshold:g}%" if accuracy is not None else "accuracy n/a"]
        measured.append(f"WAPE ratio {wape_ratio:.2f} < {ratio_threshold:g}" if wape_ratio is not None else "WAPE ratio n/a")
        return False, "; ".join(measured)
    reason = " and ".join(causes)
    trained_at = pd.Timestamp(active_model["trained_at"])
    if trained_at.tzinfo is None:
        trained_at = trained_at.tz_localize("UTC")
    if now - trained_at.to_pydatetime() < timedelta(hours=interval_hours):
        return False, f"{reason} but the model is younger than {interval_hours:g}h"
    return True, reason


def _validate_finite(value: Any, path: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise RuntimeError(f"Invalid numeric value at {path}: expected a finite number")


def validate_metrics(metrics: Any, *, expected_prediction_count: int | None = None) -> dict[str, Any]:
    if not isinstance(metrics, dict) or not metrics:
        raise RuntimeError("Metrics are missing or empty")
    test = metrics.get("test")
    if not isinstance(test, dict) or not test:
        raise RuntimeError("metrics.test is missing or empty")
    missing = [name for name in REQUIRED_METRICS if name not in test]
    if missing:
        raise RuntimeError(f"Missing required metrics: {', '.join(missing)}")
    for name in REQUIRED_METRICS:
        _validate_finite(test[name], f"metrics.test.{name}")
    for name, value in test.items():
        if value is None:
            raise RuntimeError(f"Metric is null: metrics.test.{name}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            _validate_finite(value, f"metrics.test.{name}")
    if expected_prediction_count is not None:
        reported = metrics.get("split", {}).get("test_rows")
        if reported is not None and int(reported) != expected_prediction_count:
            raise RuntimeError(
                f"Prediction count mismatch: metrics reports {reported}, file contains {expected_prediction_count}"
            )
    return test


def _validate_json_value(value: Any, path: str = "payload") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str) or not key:
                raise RuntimeError(f"Invalid payload key at {path}")
            _validate_json_value(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _validate_json_value(child, f"{path}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Non-finite value at {path}")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise RuntimeError(f"Unsupported value at {path}: {type(value).__name__}")


def validate_submission_payload(payload: Any, *, expected_prediction_count: int) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError("Submission payload must be an object")
    required = ("schema_version", "cycle_id", "client_run_id", "data_cutoff", "model", "predictions")
    missing = [field for field in required if not payload.get(field)]
    if missing:
        raise RuntimeError(f"Submission payload is missing required fields: {', '.join(missing)}")
    if not isinstance(payload["model"], dict) or not payload["model"].get("version"):
        raise RuntimeError("Submission payload model.version is required")
    predictions = payload["predictions"]
    if not isinstance(predictions, list) or len(predictions) != expected_prediction_count or not predictions:
        raise RuntimeError(
            f"Submission predictions count is invalid: expected {expected_prediction_count}, got {len(predictions) if isinstance(predictions, list) else 'non-list'}"
        )
    for index, prediction in enumerate(predictions):
        if not isinstance(prediction, dict) or not {"station_id", "target_at", "value"}.issubset(prediction):
            raise RuntimeError(f"Prediction {index} is incomplete")
        _validate_finite(prediction["value"], f"payload.predictions[{index}].value")
    _validate_json_value(payload)


def validate_submission_response(
    response: Any,
    *,
    cycle_id: str,
    expected_prediction_count: int,
) -> dict[str, Any]:
    if not isinstance(response, dict) or not response:
        raise RuntimeError("API returned an empty submission confirmation")
    status = str(response.get("status", "")).lower()
    submission_id = response.get("submission_id") or response.get("id") or response.get("operation_id")
    accepted_statuses = {"accepted", "created", "received", "success", "submitted", "ok"}
    if not submission_id and status not in accepted_statuses:
        raise RuntimeError(f"API did not accept submission: status={status or 'missing'}")
    if not submission_id:
        raise RuntimeError("API confirmation is missing submission_id")
    received = response.get("predictions_received")
    if received is not None and int(received) != expected_prediction_count:
        raise RuntimeError(
            f"API prediction confirmation mismatch: expected {expected_prediction_count}, got {received}"
        )
    contract = response.get("validated_contract")
    if isinstance(contract, dict):
        if contract.get("cycle_id") != cycle_id:
            raise RuntimeError("API validated contract belongs to a different cycle")
        contract_count = contract.get("expected_predictions")
        if contract_count is not None and int(contract_count) != expected_prediction_count:
            raise RuntimeError("API validated contract has an unexpected prediction count")
    if response.get("is_official") is False:
        raise RuntimeError("API submission was not marked official")
    return response


def persist_forecast_predictions(
    db: SupabaseRestClient,
    *,
    model_id: int,
    cycle: dict[str, Any],
    predictions: list[dict[str, Any]],
    generated_at: str,
) -> int:
    forecast_start = pd.Timestamp(cycle.get("forecast_start_at") or cycle["data_cutoff"])
    rows = []
    for prediction in predictions:
        target_at = pd.Timestamp(prediction["target_at"])
        horizon_periods = int(round((target_at - forecast_start).total_seconds() / 900)) + 1
        rows.append({
            "model_id": model_id,
            "station_id": str(prediction["station_id"]).zfill(5),
            "target_at": target_at.isoformat(),
            "generated_at": generated_at,
            "horizon_periods": horizon_periods,
            "prediction": float(prediction["value"]),
        })
    count = 0
    for start in range(0, len(rows), 500):
        count += len(db.upsert_many("prediction", rows[start : start + 500], "model_id,station_id,target_at,horizon_periods"))
    return count


def persist_execution(
    db: SupabaseRestClient,
    *,
    run_id: str,
    model_id: int,
    cycle_id: str,
    model_version: str,
    generated_at: str,
    metrics: dict[str, Any],
    prediction_count: int,
    payload_sha256: str,
) -> None:
    try:
        db.upsert_many("pipeline_execution", [{
            "run_id": run_id,
            "model_id": model_id,
            "cycle_id": cycle_id,
            "model_version": model_version,
            "generated_at": generated_at,
            "metrics": metrics,
            "prediction_count": prediction_count,
            "payload_sha256": payload_sha256,
            "status": "prepared",
        }], "run_id")
    except RuntimeError as exc:
        if "404" in str(exc):
            print("[WARNING] pipeline_execution no existe aún; se continúa y se conserva el payload en outbox")
            return
        raise


def prune_superseded_models(db: SupabaseRestClient, keep_model_id: int) -> None:
    """Bound Supabase storage: keep evaluation rows and artifact only for the current model."""
    try:
        result = db.prune_model_history(keep_model_id)
    except RuntimeError as exc:
        # Housekeeping must not block publication; the next new model prunes
        # everything older than itself again.
        if "404" in str(exc):
            print("[WARNING] prune_model_history no existe aún; aplica la migración de retención")
        else:
            print(f"[WARNING] No se pudo podar el historial de modelos: {exc}")
        return
    print(
        f"[INFO] Historial podado: métricas={result.get('metrics_deleted', 0)} "
        f"predicciones={result.get('predictions_deleted', 0)} modelos_retirados={result.get('models_retired', 0)}"
    )


def _safe_payload_path(submission_key: str) -> Path:
    digest = hashlib.sha256(submission_key.encode()).hexdigest()[:24]
    return OUTBOX_DIR / f"submission-{digest}.json"


def write_pipeline_status(status: str, detail: str) -> None:
    status_path = ARTIFACT_DIR / "pipeline-status.json"
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps({"status": status, "detail": detail, "updated_at": utc_now().isoformat()}, indent=2))
    summary_path = os.getenv("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a", encoding="utf-8") as summary:
            summary.write(f"## Pipeline: {status}\n\n{detail}\n")


def update_sync_status_state(db: SupabaseRestClient, status: str, *, error: str | None = None) -> None:
    try:
        previous = db.rows("pipeline_sync_status", select="last_updated_at", id="eq.singleton", limit="1")
        now = utc_now().isoformat()
        db.upsert_many("pipeline_sync_status", [{
            "id": "singleton",
            "status": status,
            "last_updated_at": now if status == "SUCCESS" else (previous[0].get("last_updated_at") if previous else None),
            "error": error,
            "updated_at": now,
        }], "id")
    except RuntimeError as exc:
        if "404" in str(exc):
            print("[WARNING] pipeline_sync_status no existe aún; aplique la migración para habilitar el heartbeat")
            return
        raise


def forecast_with_lags(
    model: LagEnsembleForecaster, db: SupabaseRestClient, cycle: dict[str, Any], stations: pd.DataFrame
) -> list[dict[str, Any]]:
    """Forecast the cycle targets from the demand observed up to its cutoff (ingested just before)."""
    cutoff = pd.Timestamp(cycle["data_cutoff"])
    history = paged_since(db, "demand_observation", "station_id,observed_at,demand", cutoff - HISTORY_PERIODS * PERIOD)
    history = history[history["observed_at"] <= cutoff] if not history.empty else history
    if history.empty:
        raise RuntimeError("No demand observations available before the cycle cutoff")
    targets = [pd.Timestamp(target["target_at"]) for target in cycle["targets"]]
    forecast = model.forecast(history, targets, stations["station_id"].astype(str))
    return [
        {"station_id": row["station_id"], "target_at": row["target_at"].isoformat(), "value": max(0.0, float(row["value"]))}
        for row in forecast.to_dict("records")
    ]


def forecast_open_cycle(
    api: PulsoTransmiClient, cycle: dict[str, Any], artifact: bytes, db: SupabaseRestClient | None = None
) -> list[dict[str, Any]]:
    bundle = __import__("joblib").load(ARTIFACT_DIR / "model_and_metrics.joblib")
    stations = api.stations()
    if isinstance(bundle["model"], LagEnsembleForecaster):
        if db is None:
            raise RuntimeError("The lag model needs Supabase access to read recent demand")
        return forecast_with_lags(bundle["model"], db, cycle, stations)
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
    algorithm = str(active_model.get("algorithm", "model")).removeprefix("sklearn.")
    stored_metrics = db.latest_training_metrics(int(active_model["id"]))
    metrics = stored_metrics if "test" in stored_metrics else {
        "selection": {"selected_model": algorithm},
        "test": stored_metrics,
    }
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))


def publish(
    env: dict[str, str],
    *,
    persist_evaluation: bool = True,
    db: SupabaseRestClient | None = None,
) -> dict[str, Any]:
    supabase_url = env.get("SUPABASE_URL")
    supabase_key = env.get("SUPABASE_SERVICE_ROLE_KEY") or env.get("SUPABASE_KEY")
    api_key = env.get("PULSO_API_KEY")
    api_url = env.get("PULSO_API_URL")
    if not supabase_url or not supabase_key or not api_key or not api_url:
        raise RuntimeError("SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, PULSO_API_URL and PULSO_API_KEY are required")

    metrics_path = ARTIFACT_DIR / "metrics.json"
    if not metrics_path.exists():
        raise RuntimeError(f"Metrics artifact does not exist: {metrics_path}")
    metrics = json.loads(metrics_path.read_text())
    predictions_path = ARTIFACT_DIR / "predictions.csv"
    predictions = pd.read_csv(predictions_path) if predictions_path.exists() else pd.DataFrame()
    expected_prediction_count = len(predictions)
    if expected_prediction_count == 0 and persist_evaluation:
        raise RuntimeError("Predictions artifact is empty")
    metric_values = validate_metrics(metrics, expected_prediction_count=expected_prediction_count or None)
    required_prediction_columns = {"observed_at", "station_id", "predicted_demand", "actual_demand"}
    if not predictions.empty and not required_prediction_columns.issubset(predictions.columns):
        raise RuntimeError(f"Predictions artifact is missing columns: {sorted(required_prediction_columns - set(predictions.columns))}")
    artifact_path = ARTIFACT_DIR / "model_and_metrics.joblib"
    if not artifact_path.exists():
        raise RuntimeError(f"Model artifact does not exist: {artifact_path}")
    artifact = artifact_path.read_bytes()
    artifact_sha256 = hashlib.sha256(artifact).hexdigest()
    algorithm_name = str(metrics.get("selection", {}).get("selected_model", "model"))
    version = f"{algorithm_name}-{artifact_sha256[:12]}"
    mlflow_metadata = load_metadata(ARTIFACT_DIR)
    now = utc_now()
    db = db or SupabaseRestClient(supabase_url, supabase_key)
    data_cut_id = db.latest_data_cut_id()
    existing_model = db.find_model(version)
    if existing_model:
        model_id = int(existing_model["id"])
        trained_at = str(existing_model["trained_at"])
        print(f"[inference] Reusing model version={version} id={model_id}")
        if mlflow_metadata:
            db.update("model", {"id": f"eq.{model_id}"}, {
                "mlflow_run_id": mlflow_metadata.get("run_id"),
                "mlflow_model_name": mlflow_metadata.get("registered_model_name"),
                "mlflow_model_version": mlflow_metadata.get("model_version"),
            })
            run_row = db.insert("training_run", {
                "model_id": model_id,
                "data_cut_id": data_cut_id,
                "cutoff_at": now.isoformat(),
                "parameters": {"script": str(TRAIN_SCRIPT), "artifact_sha256": artifact_sha256},
                "metrics": metrics,
                "mlflow_run_id": mlflow_metadata.get("run_id"),
                "started_at": now.isoformat(),
                "finished_at": now.isoformat(),
            })
        else:
            run_row = {"id": "reused"}
    else:
        algorithm = algorithm_name
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
            "mlflow_run_id": mlflow_metadata.get("run_id"),
            "mlflow_model_name": mlflow_metadata.get("registered_model_name"),
            "mlflow_model_version": mlflow_metadata.get("model_version"),
        })
        model_id = int(model_row["id"])
        trained_at = now.isoformat()
        run_row = db.insert("training_run", {
            "model_id": model_id,
            "data_cut_id": data_cut_id,
            "cutoff_at": now.isoformat(),
            "parameters": {"script": str(TRAIN_SCRIPT), "artifact_sha256": artifact_sha256},
            "metrics": metrics,
            "mlflow_run_id": mlflow_metadata.get("run_id"),
            "started_at": now.isoformat(),
            "finished_at": now.isoformat(),
        })

    prediction_payload = [
        {
            "model_id": model_id,
            "station_id": str(row["station_id"]).zfill(5),
            "target_at": row["observed_at"],
            "generated_at": now.isoformat(),
            "horizon_periods": int(row.get("horizon_periods", 1)),
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
                "wape": metric_values["wape"],
                "accuracy": metric_values["accuracy"],
                "mae": metric_values["mae"],
                "rmse": metric_values["rmse"],
            }
            for row in prediction_rows
        ]
        for start in range(0, len(metric_payload), 500):
            db.insert_many("monitoring_metric", metric_payload[start : start + 500])
    if not existing_model:
        prune_superseded_models(db, model_id)
    print(f"[INFO] Metrics validated: MAE={metric_values['mae']:.2f} RMSE={metric_values['rmse']:.2f} WAPE={metric_values['wape']:.4f}")

    with PulsoTransmiClient(base_url=api_url, api_key=api_key) as api:
        print("[INFO] Consultando ciclo de predicción")
        cycle = api.current_cycle()
        if cycle is None:
            detail = "There is no open forecast cycle; publication will be retried on the next scheduled run."
            print(f"[WARNING] No existe un ciclo de predicción abierto")
            print(f"[WARNING] Publicación aplazada")
            print(f"[WARNING] Estado: WAITING_FOR_OPEN_CYCLE")
            write_pipeline_status("WAITING_FOR_OPEN_CYCLE", detail)
            update_sync_status_state(db, "WAITING_FOR_OPEN_CYCLE", error=detail)
            raise WaitingForOpenCycle(detail)
        cycle_id = str(cycle.get("cycle_id") or cycle["id"])
        print(f"[INFO] Ciclo: {cycle_id}")
        print(f"[INFO] Estado del ciclo: {cycle.get('status', 'OPEN')}")
        submission_predictions = forecast_open_cycle(api, cycle, artifact, db)
        if not submission_predictions:
            raise RuntimeError("Inference generated no predictions")
        target_count = len({str(target["target_at"]) for target in cycle.get("targets", [])})
        station_count = len(api.stations())
        if target_count <= 0 or station_count <= 0:
            raise RuntimeError("Current cycle has no targets or stations")
        expected_submission_count = target_count * station_count
        if len(submission_predictions) != expected_submission_count:
            raise RuntimeError(f"Inference count mismatch: expected {expected_submission_count}, got {len(submission_predictions)}")
        # A cycle is the logical submission boundary. The model artifact can change
        # between retries, but that must never create a second submission for it.
        submission_key = cycle_id
        client_run_id = hashlib.sha256(submission_key.encode()).hexdigest()[:32]
        payload = {
            "schema_version": "1.0",
            "cycle_id": cycle_id,
            "client_run_id": client_run_id,
            "data_cutoff": cycle["data_cutoff"],
            "model": {"version": version, "trained_at": trained_at, "training_data_end": cycle["data_cutoff"], "git_commit": os.getenv("GITHUB_SHA")},
            "predictions": submission_predictions,
        }
        validate_submission_payload(payload, expected_prediction_count=expected_submission_count)
        payload_sha256 = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        metric_names = ", ".join(sorted(metric_values))
        print(f"[INFO] Predicciones generadas/preparadas: {len(submission_predictions)}/{expected_submission_count}")
        print(f"[INFO] Métricas calculadas: {len(metric_values)} ({metric_names})")
        print("[INFO] Validación del payload: OK")
        try:
            existing = db.rows("pipeline_execution", select="status,payload_sha256,model_version,api_submission_id", run_id=f"eq.{client_run_id}", limit="1")
        except RuntimeError as exc:
            if "404" not in str(exc):
                raise
            existing = []
        if submitted_with_other_content(existing[0] if existing else None, payload_sha256):
            # Keep the confirmed record intact; a new model submits from the next cycle.
            update_sync_status_state(db, "SUCCESS")
            raise CycleAlreadySubmitted(f"cycle {cycle_id} already confirmed with model {existing[0].get('model_version')}; model {version} submits from the next cycle")
        persist_execution(
            db,
            run_id=client_run_id,
            model_id=model_id,
            cycle_id=cycle_id,
            model_version=version,
            generated_at=now.isoformat(),
            metrics=metrics,
            prediction_count=len(submission_predictions),
            payload_sha256=payload_sha256,
        )
        OUTBOX_DIR.mkdir(parents=True, exist_ok=True)
        outbox_path = _safe_payload_path(submission_key)
        outbox_path.write_text(json.dumps({"idempotency_key": submission_key, "run_id": client_run_id, "model_version": version, "metrics": metrics, "payload": payload}, indent=2, allow_nan=False))
        print("[INFO] Generando submission")
        print("[INFO] Publicando resultados")
        try:
            submission = validate_submission_response(
                api.submit(payload, idempotency_key=submission_key),
                cycle_id=cycle_id,
                expected_prediction_count=expected_submission_count,
            )
        except Exception as exc:
            if is_idempotency_conflict(exc):
                db.update("pipeline_execution", {"run_id": f"eq.{client_run_id}"}, {
                    "status": "failed",
                    "error": f"cycle already submitted by an earlier run; model {version} submits from the next cycle",
                })
                update_sync_status_state(db, "SUCCESS")
                raise CycleAlreadySubmitted(f"cycle {cycle_id} was already submitted with different content; model {version} submits from the next cycle") from exc
            db.update("pipeline_execution", {"run_id": f"eq.{client_run_id}"}, {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            })
            update_sync_status_state(db, "FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        print(
            f"[INFO] API confirmed receipt: submission_id={submission.get('submission_id') or submission.get('id')} "
            f"cycle={cycle_id} model={version} predictions={len(submission_predictions)}"
        )
        try:
            persisted = persist_forecast_predictions(
                db,
                model_id=model_id,
                cycle=cycle,
                predictions=submission_predictions,
                generated_at=now.isoformat(),
            )
            if persisted != expected_submission_count:
                raise RuntimeError(f"Supabase prediction confirmation mismatch: expected {expected_submission_count}, got {persisted}")
            db.update("pipeline_execution", {"run_id": f"eq.{client_run_id}"}, {
                "status": "confirmed",
                "api_submission_id": submission.get("submission_id") or submission.get("id"),
                "api_response": submission,
                "confirmed_at": utc_now().isoformat(),
            })
        except Exception as exc:
            db.update("pipeline_execution", {"run_id": f"eq.{client_run_id}"}, {
                "status": "partial_failure",
                "api_submission_id": submission.get("submission_id") or submission.get("id"),
                "api_response": submission,
                "error": f"{type(exc).__name__}: {exc}",
            })
            update_sync_status_state(db, "FAILED", error=f"{type(exc).__name__}: {exc}")
            raise
        outbox_path.unlink(missing_ok=True)
        update_sync_status_state(db, "SUCCESS")
        print(f"[INFO] Supabase confirmed forecast predictions: {persisted}/{expected_submission_count}")
    return {"model_id": model_id, "training_run_id": run_row["id"], "submission": submission}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--publish-only", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    args = parser.parse_args()
    env = load_env()
    print("[INFO] Pipeline iniciado")
    supabase_url = env.get("SUPABASE_URL")
    supabase_key = env.get("SUPABASE_SERVICE_ROLE_KEY") or env.get("SUPABASE_KEY")
    if not supabase_url or not supabase_key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")
    try:
        with SupabaseRestClient(supabase_url, supabase_key) as db:
            publish_only = args.publish_only
            if not publish_only and not args.force_retrain:
                active_model = db.active_model()
                measured = recent_accuracy(db, int(active_model["id"])) if usable_model(active_model) else {"accuracy": None, "source": "unavailable", "rows": 0}
                if measured["accuracy"] is not None:
                    print(f"[INFO] Recent accuracy {measured['accuracy']:.2f}% (source={measured['source']}, rows={measured['rows']})")
                wape_ratio = None
                if measured.get("wape") is not None:
                    baseline = baseline_wape(db.latest_training_metrics(int(active_model["id"])))
                    wape_ratio = measured["wape"] / baseline if baseline > 0 else None
                    if wape_ratio is not None:
                        print(f"[INFO] Recent WAPE {measured['wape']:.4f} vs test {baseline:.4f} (ratio {wape_ratio:.2f})")
                retrain, reason = retrain_decision(active_model, measured["accuracy"], utc_now(), wape_ratio=wape_ratio)
                publish_only = not retrain
                print(f"[INFO] {'Retraining' if retrain else 'Skipping retraining'}: {reason}")
            if publish_only:
                restore_active_artifacts(db)
                print("[INFO] Active model artifact restored from Supabase")
                result = publish(env, persist_evaluation=False, db=db)
            else:
                run_training()
                print("[INFO] Training and inference artifacts generated")
                result = publish(env, db=db)
        print(json.dumps(result, indent=2, default=str))
        write_pipeline_status("SUCCESS", "Ingesta, inferencia, métricas y publicación confirmadas.")
        print("[INFO] PIPELINE SUCCESS")
    except CycleAlreadySubmitted as exc:
        write_pipeline_status("SUCCESS", f"Ciclo ya enviado: {exc}")
        print(f"[WARNING] PIPELINE CYCLE_ALREADY_SUBMITTED: {exc}")
    except WaitingForOpenCycle:
        print("[WARNING] PIPELINE WAITING_FOR_OPEN_CYCLE", file=sys.stderr)
        raise SystemExit(2)
    except Exception as exc:
        # Never include environment values or request headers in pipeline logs.
        write_pipeline_status("FAILED", f"{type(exc).__name__}: {exc}")
        print(f"[ERROR] PIPELINE FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
