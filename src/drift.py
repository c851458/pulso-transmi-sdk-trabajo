from __future__ import annotations

import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from src.ingest import load_env


LOG = logging.getLogger("pulso.drift")
REFERENCE_DAYS = int(os.getenv("DRIFT_REFERENCE_DAYS", "7"))
CURRENT_DAYS = int(os.getenv("DRIFT_CURRENT_DAYS", "7"))
PSI_THRESHOLD = float(os.getenv("DRIFT_PSI_THRESHOLD", "0.20"))
DRIFTED_FEATURES_REQUIRED = int(os.getenv("DRIFTED_FEATURES_REQUIRED", "2"))
PERFORMANCE_RATIO_THRESHOLD = float(os.getenv("DRIFT_PERFORMANCE_RATIO", "1.25"))
MIN_ROWS = int(os.getenv("DRIFT_MIN_ROWS", "96"))
RETRAIN_COOLDOWN_HOURS = int(os.getenv("RETRAIN_COOLDOWN_HOURS", "24"))
EPSILON = 1e-6


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
        with urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
        if not isinstance(result, list):
            raise RuntimeError(f"Unexpected Supabase response for {table}")
        return result

    def rows(self, table: str, **params: str) -> list[dict[str, Any]]:
        return self.request("GET", table, params=params)

    def insert(self, table: str, payload: dict[str, Any]) -> dict[str, Any]:
        rows = self.request("POST", table, payload)
        if not rows:
            raise RuntimeError(f"Supabase did not return the inserted {table}")
        return rows[0]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def paged_since(db: SupabaseRestClient, table: str, select: str, start: pd.Timestamp) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    page_size = 1_000
    for offset in range(0, 100_000, page_size):
        page = db.rows(
            table,
            select=select,
            observed_at=f"gte.{start.isoformat()}",
            order="observed_at.asc",
            limit=str(page_size),
            offset=str(offset),
        )
        rows.extend(page)
        if len(page) < page_size:
            break
    else:
        raise RuntimeError(f"Drift query pagination limit reached for {table}")
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
    return frame


def psi(reference: pd.Series, current: pd.Series) -> float:
    reference = pd.to_numeric(reference, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    current = pd.to_numeric(current, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if reference.empty or current.empty:
        return float("nan")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0] = -np.inf
    edges[-1] = np.inf
    reference_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    current_pct = np.histogram(current, bins=edges)[0] / len(current)
    reference_pct = np.maximum(reference_pct, EPSILON)
    current_pct = np.maximum(current_pct, EPSILON)
    return float(np.sum((current_pct - reference_pct) * np.log(current_pct / reference_pct)))


def latest_model(db: SupabaseRestClient) -> dict[str, Any] | None:
    rows = db.rows("model", select="id,version,trained_at,status", order="trained_at.desc", limit="1")
    return rows[0] if rows else None


def latest_training_metrics(db: SupabaseRestClient, model_id: int | None) -> dict[str, Any]:
    params = {"select": "metrics", "order": "started_at.desc", "limit": "1"}
    if model_id is not None:
        params["model_id"] = f"eq.{model_id}"
    rows = db.rows("training_run", **params)
    return rows[0].get("metrics") or {} if rows else {}


def evaluate(env: dict[str, str]) -> dict[str, Any]:
    db = SupabaseRestClient(env["SUPABASE_URL"], env["SUPABASE_SERVICE_ROLE_KEY"])
    latest_rows = db.rows("demand_observation", select="observed_at", order="observed_at.desc", limit="1")
    if not latest_rows:
        raise RuntimeError("No demand observations available for drift evaluation")
    latest = pd.Timestamp(latest_rows[0]["observed_at"])
    current_start = latest - timedelta(days=CURRENT_DAYS)
    reference_start = current_start - timedelta(days=REFERENCE_DAYS)
    demand = paged_since(db, "demand_observation", "station_id,observed_at,demand", reference_start)
    context = paged_since(db, "context_observation", "observed_at,event_intensity,rain_forecast,rain_mm,temperature_c,temperature_forecast", reference_start)
    reference_end = current_start
    reference_demand = demand[(demand["observed_at"] >= reference_start) & (demand["observed_at"] < reference_end)]
    current_demand = demand[demand["observed_at"] >= current_start]
    reference_context = context[(context["observed_at"] >= reference_start) & (context["observed_at"] < reference_end)]
    current_context = context[context["observed_at"] >= current_start]
    if len(reference_demand) < MIN_ROWS or len(current_demand) < MIN_ROWS:
        raise RuntimeError(f"Insufficient data for drift: reference={len(reference_demand)} current={len(current_demand)}")

    numeric_features = ["demand", "event_intensity", "rain_forecast", "rain_mm", "temperature_c", "temperature_forecast"]
    feature_scores: dict[str, float] = {}
    feature_alerts: list[str] = []
    for feature in numeric_features:
        reference_frame = reference_demand if feature == "demand" else reference_context
        current_frame = current_demand if feature == "demand" else current_context
        score = psi(reference_frame[feature], current_frame[feature])
        feature_scores[feature] = score
        if not math.isnan(score) and score >= PSI_THRESHOLD:
            feature_alerts.append(feature)

    model = latest_model(db)
    model_id = int(model["id"]) if model else None
    training_metrics = latest_training_metrics(db, model_id)
    baseline_wape = float(training_metrics.get("wape", 0.0) or 0.0)
    performance = {"baseline_wape": baseline_wape, "recent_wape": None, "ratio": None, "alert": False}
    if model_id is not None and baseline_wape > 0:
        prediction_rows = db.rows(
            "prediction",
            select="prediction,actual_value,target_at",
            model_id=f"eq.{model_id}",
            actual_value="not.is.null",
            target_at=f"gte.{current_start.isoformat()}",
            limit="10000",
        )
        if len(prediction_rows) >= MIN_ROWS:
            prediction_frame = pd.DataFrame(prediction_rows)
            actual = pd.to_numeric(prediction_frame["actual_value"], errors="coerce")
            predicted = pd.to_numeric(prediction_frame["prediction"], errors="coerce")
            recent_wape = float((actual - predicted).abs().sum() / max(actual.sum(), 1.0))
            ratio = recent_wape / baseline_wape
            performance = {"baseline_wape": baseline_wape, "recent_wape": recent_wape, "ratio": ratio, "alert": ratio >= PERFORMANCE_RATIO_THRESHOLD}

    drifted_count = len(feature_alerts)
    drift_alert = drifted_count >= DRIFTED_FEATURES_REQUIRED or performance["alert"]
    cooldown = False
    if model:
        trained_at = pd.Timestamp(model["trained_at"])
        cooldown = trained_at >= (utc_now() - timedelta(hours=RETRAIN_COOLDOWN_HOURS))
    retrain = drift_alert and not cooldown
    reason = "; ".join([
        f"features={','.join(feature_alerts) or 'none'}",
        f"performance_alert={performance['alert']}",
        f"cooldown={cooldown}",
    ])
    report = {
        "checked_at": utc_now().isoformat(),
        "model_id": model_id,
        "model_version": model.get("version") if model else None,
        "reference_window": {"start": reference_start.isoformat(), "end": reference_end.isoformat(), "rows": len(reference_demand)},
        "current_window": {"start": current_start.isoformat(), "end": latest.isoformat(), "rows": len(current_demand)},
        "thresholds": {"psi": PSI_THRESHOLD, "drifted_features_required": DRIFTED_FEATURES_REQUIRED, "performance_ratio": PERFORMANCE_RATIO_THRESHOLD, "cooldown_hours": RETRAIN_COOLDOWN_HOURS},
        "feature_psi": feature_scores,
        "drifted_features": feature_alerts,
        "performance": performance,
        "drift_alert": drift_alert,
        "retrain": retrain,
        "reason": reason,
    }
    db.insert("model_drift_check", report)
    return report


def write_output(report: dict[str, Any]) -> None:
    output_path = os.getenv("GITHUB_OUTPUT")
    if output_path:
        with open(output_path, "a", encoding="utf-8") as output:
            output.write(f"drift_alert={'true' if report['drift_alert'] else 'false'}\n")
            output.write(f"retrain={'true' if report['retrain'] else 'false'}\n")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    LOG.info("Inicio de evaluación de drift")
    env = load_env()
    required = ("SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY")
    missing = [key for key in required if not env.get(key)]
    if missing:
        raise RuntimeError(f"Missing required variables: {', '.join(missing)}")
    report = evaluate(env)
    LOG.info("Ventanas: referencia=%d actual=%d", report["reference_window"]["rows"], report["current_window"]["rows"])
    LOG.info("PSI por variable: %s", json.dumps(report["feature_psi"], sort_keys=True))
    LOG.info("Drift alert=%s retrain=%s reason=%s", report["drift_alert"], report["retrain"], report["reason"])
    write_output(report)
    LOG.info("Evaluación de drift finalizada")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("Error evaluando drift; la próxima ejecución podrá reintentarlo")
        raise
