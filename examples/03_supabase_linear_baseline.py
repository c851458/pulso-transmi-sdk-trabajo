from __future__ import annotations

import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.mlflow_tracking import log_training_run
from src.supabase_resilience import SupabaseRestClient


TABLE_PAGE_SIZE = 1000
TRAIN_FRACTION = 0.8
RANDOM_STATE = 42
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


def fetch_dataset(client: SupabaseRestClient) -> pd.DataFrame:
    stations = pd.DataFrame(client.get_all(
        "station", "station_id,station_name,corridor,latitude,longitude"
    ))
    observations = pd.DataFrame(client.get_all(
        "demand_observation", "station_id,observed_at,demand"
    ))
    context = pd.DataFrame(client.get_all(
        "context_observation",
        "observed_at,event_intensity,rain_forecast,rain_mm,temperature_c,temperature_forecast",
    ))
    observations["observed_at"] = pd.to_datetime(
        observations["observed_at"], utc=True
    )
    context["observed_at"] = pd.to_datetime(context["observed_at"], utc=True)
    dataset = (
        observations.merge(
            context, on="observed_at", how="inner", validate="many_to_one"
        )
        .merge(stations, on="station_id", how="inner", validate="many_to_one")
        .sort_values(["observed_at", "station_id"])
        .reset_index(drop=True)
    )
    if len(dataset) != len(observations):
        raise ValueError("The joins changed the observation row count")
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    dataset = dataset.drop_duplicates(["station_id", "observed_at"], keep="last")
    if dataset["demand"].isna().any() or (dataset["demand"] < 0).any():
        raise ValueError("Target demand contains invalid values")
    return dataset


def add_features(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy()
    minutes = result["observed_at"].dt.hour * 60 + result["observed_at"].dt.minute
    result["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    result["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
    result["weekday_sin"] = np.sin(2 * np.pi * result["observed_at"].dt.dayofweek / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * result["observed_at"].dt.dayofweek / 7)
    return result


def feature_columns() -> tuple[list[str], list[str]]:
    categorical = ["station_id", "corridor"]
    numeric = [
        "latitude",
        "longitude",
        "event_intensity",
        "rain_forecast",
        "rain_mm",
        "temperature_c",
        "temperature_forecast",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
    ]
    return categorical, numeric


def build_preprocessor(*, dense: bool = False) -> ColumnTransformer:
    categorical, numeric = feature_columns()
    preprocessor = ColumnTransformer(
        [
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
            ]), categorical),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]), numeric),
        ],
        remainder="drop",
    )
    return preprocessor


def build_models() -> dict[str, Pipeline]:
    return {
        "baseline_linear": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", LinearRegression()),
        ]),
        "robust_ridge": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", Ridge(alpha=10.0)),
        ]),
        "random_forest": Pipeline([
            ("preprocess", build_preprocessor()),
            ("model", RandomForestRegressor(
                n_estimators=80,
                max_depth=12,
                min_samples_leaf=10,
                max_features=0.5,
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocess", build_preprocessor(dense=True)),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=31,
                min_samples_leaf=30,
                l2_regularization=1.0,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=20,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def evaluate_model(name: str, model: Pipeline, train: pd.DataFrame, test: pd.DataFrame, predictors: list[str], target: str) -> tuple[dict[str, object], Pipeline, np.ndarray]:
    started = time.perf_counter()
    cv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_validate(
        model,
        train[predictors],
        train[target],
        cv=cv,
        scoring={"mae": "neg_mean_absolute_error", "rmse": "neg_root_mean_squared_error", "r2": "r2"},
        n_jobs=None,
    )
    model.fit(train[predictors], train[target])
    predictions = np.maximum(0.0, model.predict(test[predictors]))
    actual = test[target].to_numpy()
    absolute_error = np.abs(actual - predictions)
    elapsed = time.perf_counter() - started
    result = {
        "model": name,
        "cv_mae_mean": float(-cv_scores["test_mae"].mean()),
        "cv_mae_std": float(cv_scores["test_mae"].std()),
        "cv_rmse_mean": float(-cv_scores["test_rmse"].mean()),
        "cv_rmse_std": float(cv_scores["test_rmse"].std()),
        "cv_r2_mean": float(cv_scores["test_r2"].mean()),
        "cv_r2_std": float(cv_scores["test_r2"].std()),
        "mae": float(mean_absolute_error(actual, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predictions))),
        "r2": float(r2_score(actual, predictions)),
        "wape": float(absolute_error.sum() / max(actual.sum(), 1.0)),
        "training_seconds": float(elapsed),
        "stability_score": float(cv_scores["test_mae"].std() / max(-cv_scores["test_mae"].mean(), 1.0)),
    }
    return result, model, predictions


def main() -> None:
    env = load_env()
    url = env.get("SUPABASE_URL")
    key = env.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    with SupabaseRestClient(url, key) as client:
        dataset = add_features(fetch_dataset(client))
    split_at = int(len(dataset) * TRAIN_FRACTION)
    train = dataset.iloc[:split_at].copy()
    test = dataset.iloc[split_at:].copy()
    target = "demand"
    predictors = [
        "station_id",
        "corridor",
        "latitude",
        "longitude",
        "event_intensity",
        "rain_forecast",
        "rain_mm",
        "temperature_c",
        "temperature_forecast",
        "hour_sin",
        "hour_cos",
        "weekday_sin",
        "weekday_cos",
    ]
    model_results = []
    fitted_models = {}
    test_predictions = {}
    for name, candidate in build_models().items():
        result, fitted, candidate_predictions = evaluate_model(name, candidate, train, test, predictors, target)
        model_results.append(result)
        fitted_models[name] = fitted
        test_predictions[name] = candidate_predictions
    comparison = pd.DataFrame(model_results).sort_values(["cv_mae_mean", "stability_score"])
    selected_name = str(comparison.iloc[0]["model"])
    model = fitted_models[selected_name]
    predictions = test_predictions[selected_name]
    selected_result = next(result for result in model_results if result["model"] == selected_name)
    metrics = {
        "dataset": {
            "table": "demand_observation + context_observation + station",
            "observations": int(len(dataset)),
            "variables_after_join": int(dataset.shape[1]),
            "target": target,
            "predictors": predictors,
        },
        "split": {
            "method": "chronological",
            "train_fraction": TRAIN_FRACTION,
            "test_fraction": 1 - TRAIN_FRACTION,
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "random_state": None,
            "reason": "time-series leakage prevention",
        },
        "selection": {"criterion": "lowest temporal CV MAE, then stability", "selected_model": selected_name},
        "model_comparison": model_results,
        "test": {
            **selected_result,
            "mse": float(mean_squared_error(test[target], predictions)),
            "accuracy": 100 * max(0.0, 1.0 - selected_result["wape"]),
        },
    }
    exploration = {
        "tables": ["station", "demand_observation", "context_observation"],
        "shape": {"rows": int(dataset.shape[0]), "columns": int(dataset.shape[1])},
        "dtypes": {column: str(dtype) for column, dtype in dataset.dtypes.items()},
        "nulls": {column: int(value) for column, value in dataset.isna().sum().items()},
        "duplicate_rows": int(dataset.duplicated().sum()),
        "duplicate_station_timestamp_keys": int(
            dataset.duplicated(["station_id", "observed_at"]).sum()
        ),
        "target_distribution": {
            "count": int(dataset[target].count()),
            "mean": float(dataset[target].mean()),
            "std": float(dataset[target].std()),
            "min": float(dataset[target].min()),
            "median": float(dataset[target].median()),
            "max": float(dataset[target].max()),
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
            "target": target,
            "predictors": predictors,
            "split": metrics["split"],
        },
        ARTIFACT_DIR / "model_and_metrics.joblib",
    )
    comparison.to_csv(ARTIFACT_DIR / "model_comparison.csv", index=False)
    pd.DataFrame(
        {
            "observed_at": test["observed_at"].astype(str),
            "station_id": test["station_id"],
            "actual_demand": test[target],
            "predicted_demand": predictions,
        }
    ).to_csv(ARTIFACT_DIR / "predictions.csv", index=False)
    tracking_params = {
        "model_name": selected_name,
        "target": target,
        "predictors": predictors,
        "candidate_models": list(build_models().keys()),
        "train_fraction": TRAIN_FRACTION,
        "test_fraction": 1 - TRAIN_FRACTION,
        "cv_splits": 5,
        "random_state": RANDOM_STATE,
        "dataset_tables": metrics["dataset"]["table"],
        "dataset_rows": metrics["dataset"]["observations"],
        "window_start": dataset["observed_at"].min().isoformat(),
        "window_end": dataset["observed_at"].max().isoformat(),
    }
    tracking_tags = {
        "project": "pulso-transmi",
        "model_type": selected_name,
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
