from __future__ import annotations

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.lag_model import HORIZONS, LagEnsembleForecaster, demand_matrix
from src.mlflow_tracking import log_training_run
from src.supabase_resilience import SupabaseRestClient


MODEL_NAME = "lag_adaptive_ensemble"
# The temporal test covers the latest days, the ones closest to the regime the model will forecast.
TEST_DAYS = int(os.getenv("TEST_DAYS", "3"))
# Recency half-life of the boosted correction: data this many days old weighs half.
RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "7"))
ARTIFACT_DIR = Path("artifacts/baseline")


def load_env(path: Path = Path(".env")) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text().splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    values.update({key: value for key, value in os.environ.items() if key.startswith("SUPABASE_")})
    return values


def fetch_observations(client: SupabaseRestClient) -> pd.DataFrame:
    """Demand history. Context is not used: the API publishes none after the starter
    dataset, so it would be stale at every forecast."""
    observations = pd.DataFrame(client.get_all(
        "demand_observation", "station_id,observed_at,demand", order="observed_at.asc,station_id.asc"
    ))
    observations["station_id"] = observations["station_id"].astype(str).str.zfill(5)
    observations["observed_at"] = pd.to_datetime(observations["observed_at"], utc=True)
    observations = observations.drop_duplicates(["station_id", "observed_at"], keep="last")
    if observations["demand"].isna().any() or (observations["demand"] < 0).any():
        raise ValueError("Target demand contains invalid values")
    return observations.sort_values(["observed_at", "station_id"]).reset_index(drop=True)


def score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "wape": float(np.abs(actual - predicted).sum() / max(float(actual.sum()), 1.0)),
    }


def evaluate(model: LagEnsembleForecaster, observations: pd.DataFrame, test_start: pd.Timestamp) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    """Rolling-origin test: every target is forecast from data known ``horizon`` periods earlier,
    with the correction model fitted only on data before ``test_start``."""
    demand = demand_matrix(observations)
    predictions = []
    member_rows = []
    for horizon in HORIZONS:
        actual = demand.loc[test_start:]
        frames = {**model.member_forecasts(demand, horizon), MODEL_NAME: model.predict_matrix(demand, horizon)}
        for name, frame in frames.items():
            long = pd.DataFrame({"actual_demand": actual.stack(), "predicted_demand": frame.loc[test_start:].stack()}).dropna()
            long["predicted_demand"] = long["predicted_demand"].clip(lower=0.0)
            member_rows.append({"model": name, "horizon_periods": horizon, **score(long["actual_demand"].to_numpy(), long["predicted_demand"].to_numpy())})
            if name == MODEL_NAME:
                predictions.append(long.assign(horizon_periods=horizon))
    test = pd.concat(predictions).reset_index()
    test.columns = ["observed_at", "station_id", *test.columns[2:]]
    return test, member_rows


def main() -> None:
    env = load_env()
    url = env.get("SUPABASE_URL")
    key = env.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    started = time.perf_counter()
    with SupabaseRestClient(url, key) as client:
        observations = fetch_observations(client)
    test_start = observations["observed_at"].max().floor("D") - pd.Timedelta(days=TEST_DAYS - 1)
    train = observations[observations["observed_at"] < test_start]
    evaluator = LagEnsembleForecaster(half_life_days=RECENCY_HALF_LIFE_DAYS).fit(train)
    test, member_rows = evaluate(evaluator, observations, test_start)
    actual = test["actual_demand"].to_numpy()
    predicted = test["predicted_demand"].to_numpy()
    overall = score(actual, predicted)
    # Daily folds of the test window: spread across regimes, and the most recent day.
    folds = [score(day["actual_demand"].to_numpy(), day["predicted_demand"].to_numpy()) for _, day in test.groupby(test["observed_at"].dt.floor("D"))]
    fold_mae = np.array([fold["mae"] for fold in folds])
    fold_rmse = np.array([fold["rmse"] for fold in folds])
    # The deployed model refits the correction on every observation, test days included.
    model = LagEnsembleForecaster(half_life_days=RECENCY_HALF_LIFE_DAYS).fit(observations)
    elapsed = time.perf_counter() - started

    comparison = (
        pd.DataFrame(member_rows)
        .groupby("model", as_index=False)[["mae", "rmse", "wape"]].mean()
        .sort_values("wape")
    )
    per_horizon = {f"wape_h{row['horizon_periods']}": row["wape"] for row in member_rows if row["model"] == MODEL_NAME}
    metrics = {
        "dataset": {
            "table": "demand_observation",
            "observations": int(len(observations)),
            "window_start": observations["observed_at"].min().isoformat(),
            "window_end": observations["observed_at"].max().isoformat(),
            "target": "demand",
            "horizons": list(HORIZONS),
        },
        "split": {
            "method": "chronological, rolling origin per horizon",
            "test_start": test_start.isoformat(),
            "test_days": TEST_DAYS,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "random_state": None,
            "reason": "time-series leakage prevention",
        },
        "selection": {
            "criterion": "members weighted per station by their MAE over the last error window",
            "selected_model": MODEL_NAME,
        },
        "model_comparison": comparison.to_dict("records"),
        "test": {
            "model": MODEL_NAME,
            **overall,
            **per_horizon,
            "mse": float(overall["rmse"] ** 2),
            "r2": float(r2_score(actual, predicted)),
            "accuracy": 100 * max(0.0, 1.0 - overall["wape"]),
            "cv_mae_mean": float(fold_mae.mean()),
            "cv_mae_std": float(fold_mae.std()),
            "cv_rmse_mean": float(fold_rmse.mean()),
            "cv_rmse_std": float(fold_rmse.std()),
            "cv_recent_mae": float(fold_mae[-1]),
            "cv_worst_mae": float(fold_mae.max()),
            "stability_score": float(fold_mae.std() / max(fold_mae.mean(), 1.0)),
            "training_seconds": float(elapsed),
        },
    }
    exploration = {
        "tables": ["demand_observation"],
        "shape": {"rows": int(observations.shape[0]), "columns": int(observations.shape[1])},
        "stations": int(observations["station_id"].nunique()),
        "duplicate_station_timestamp_keys": 0,
        "target_distribution": {
            "count": int(observations["demand"].count()),
            "mean": float(observations["demand"].mean()),
            "std": float(observations["demand"].std()),
            "min": float(observations["demand"].min()),
            "median": float(observations["demand"].median()),
            "max": float(observations["demand"].max()),
        },
    }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "exploration.json").write_text(json.dumps(exploration, indent=2))
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))
    dump(
        {
            "model": model,
            "metrics": metrics,
            "exploration": exploration,
            "target": "demand",
            "split": metrics["split"],
        },
        ARTIFACT_DIR / "model_and_metrics.joblib",
    )
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    test.assign(observed_at=test["observed_at"].map(pd.Timestamp.isoformat))[
        ["observed_at", "station_id", "horizon_periods", "actual_demand", "predicted_demand"]
    ].to_csv(ARTIFACT_DIR / "predictions.csv", index=False)
    tracking_params = {
        "model_name": MODEL_NAME,
        "target": "demand",
        "horizons": list(HORIZONS),
        "candidate_models": comparison["model"].tolist(),
        "test_days": TEST_DAYS,
        "recency_half_life_days": RECENCY_HALF_LIFE_DAYS,
        "error_window": model.error_window,
        "weight_power": model.weight_power,
        "dataset_tables": metrics["dataset"]["table"],
        "dataset_rows": metrics["dataset"]["observations"],
        "window_start": metrics["dataset"]["window_start"],
        "window_end": metrics["dataset"]["window_end"],
    }
    tracking_tags = {
        "project": "pulso-transmi",
        "model_type": MODEL_NAME,
        "environment": os.getenv("MLFLOW_ENVIRONMENT", "development"),
        "dataset_version": os.getenv("DATASET_VERSION", "supabase-current"),
        "training_type": os.getenv("TRAINING_TYPE", "baseline-or-drift-retrain"),
        "git_commit": os.getenv("GITHUB_SHA", "unknown"),
    }
    metadata = log_training_run(
        model=model,
        metrics=metrics,
        artifact_dir=ARTIFACT_DIR,
        params=tracking_params,
        tags=tracking_tags,
    )
    print(json.dumps({"mlflow": metadata}, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved results to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
