"""Recent accuracy of a model, shared by the pipeline and drift retraining gates."""

from __future__ import annotations

import math
import os
from datetime import timedelta
from typing import Any

import pandas as pd

from src.supabase_resilience import SupabaseRestClient


WINDOW_DAYS = int(os.getenv("ACCURACY_WINDOW_DAYS", "7"))
MIN_ROWS = int(os.getenv("ACCURACY_MIN_ROWS", "96"))
PAGE_SIZE = 1_000


def accuracy_from_wape(wape: float) -> float:
    return 100 * max(0.0, 1.0 - wape)


def baseline_wape(metrics: dict[str, Any]) -> float:
    """Read the temporal-test WAPE persisted by the training pipeline."""
    test_metrics = metrics.get("test")
    value = test_metrics.get("wape") if isinstance(test_metrics, dict) else metrics.get("wape")
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) and value >= 0 else 0.0


def wape(actual: pd.Series, predicted: pd.Series) -> float | None:
    actual = pd.to_numeric(actual, errors="coerce")
    predicted = pd.to_numeric(predicted, errors="coerce")
    valid = actual.notna() & predicted.notna()
    if valid.sum() == 0:
        return None
    value = float((actual[valid] - predicted[valid]).abs().sum() / max(float(actual[valid].sum()), 1.0))
    return value if math.isfinite(value) else None


def _paged(db: SupabaseRestClient, table: str, **params: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for page in range(100):
        batch = db.rows(table, **params, limit=str(PAGE_SIZE), offset=str(page * PAGE_SIZE))
        rows.extend(batch)
        if len(batch) < PAGE_SIZE:
            return rows
    raise RuntimeError(f"Accuracy query pagination limit reached for {table}")


def forecast_accuracy_frame(forecasts: pd.DataFrame, observations: pd.DataFrame) -> pd.DataFrame:
    """Match submitted forecasts with the demand later observed for the same station and time."""
    if forecasts.empty or observations.empty:
        return pd.DataFrame(columns=["prediction", "actual"])
    left = forecasts.assign(station_id=forecasts["station_id"].astype(str).str.zfill(5), at=pd.to_datetime(forecasts["target_at"], utc=True))
    right = observations.assign(station_id=observations["station_id"].astype(str).str.zfill(5), at=pd.to_datetime(observations["observed_at"], utc=True))
    matched = left.merge(right[["station_id", "at", "demand"]], on=["station_id", "at"], how="inner")
    return matched.rename(columns={"demand": "actual"})[["prediction", "actual"]]


def recent_accuracy(db: SupabaseRestClient, model_id: int) -> dict[str, Any]:
    """Accuracy = 100 x (1 - WAPE) of the model over the most recent observed window.

    Prefers submitted forecasts compared with the demand observed afterwards ("forecast").
    Until enough of those exist, falls back to the model's evaluation predictions whose
    targets fall in the latest observed window ("evaluation"), the same data drift uses.
    """
    latest_rows = db.rows("demand_observation", select="observed_at", order="observed_at.desc", limit="1")
    if not latest_rows:
        return {"accuracy": None, "wape": None, "rows": 0, "source": "unavailable"}
    latest = pd.Timestamp(latest_rows[0]["observed_at"])
    start = latest - timedelta(days=WINDOW_DAYS)
    window = f"(target_at.gte.{start.isoformat()},target_at.lte.{latest.isoformat()})"

    forecasts = pd.DataFrame(_paged(
        db, "prediction", select="station_id,target_at,prediction",
        model_id=f"eq.{model_id}", actual_value="is.null", **{"and": window}, order="id.asc",
    ))
    if len(forecasts) >= MIN_ROWS:
        observations = pd.DataFrame(_paged(
            db, "demand_observation", select="station_id,observed_at,demand",
            **{"and": f"(observed_at.gte.{start.isoformat()},observed_at.lte.{latest.isoformat()})"}, order="id.asc",
        ))
        matched = forecast_accuracy_frame(forecasts, observations)
        value = wape(matched["actual"], matched["prediction"]) if len(matched) >= MIN_ROWS else None
        if value is not None:
            return {"accuracy": accuracy_from_wape(value), "wape": value, "rows": len(matched), "source": "forecast"}

    evaluation = pd.DataFrame(_paged(
        db, "prediction", select="prediction,actual_value",
        model_id=f"eq.{model_id}", actual_value="not.is.null", target_at=f"gte.{start.isoformat()}", order="id.asc",
    ))
    if len(evaluation) >= MIN_ROWS:
        value = wape(evaluation["actual_value"], evaluation["prediction"])
        if value is not None:
            return {"accuracy": accuracy_from_wape(value), "wape": value, "rows": len(evaluation), "source": "evaluation"}
    return {"accuracy": None, "wape": None, "rows": 0, "source": "unavailable"}
