from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TABLE_PAGE_SIZE = 1000
TRAIN_FRACTION = 0.8
RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts/baseline")


def load_env(path: Path = Path(".env")) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


class SupabaseRestClient:
    def __init__(self, url: str, key: str) -> None:
        self.base_url = url.rstrip("/") + "/rest/v1"
        self.headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
        }

    def get_page(self, table: str, select: str, offset: int) -> list[dict[str, object]]:
        query = urlencode(
            {"select": select, "limit": TABLE_PAGE_SIZE, "offset": offset}
        )
        request = Request(
            f"{self.base_url}/{table}?{query}", headers=self.headers
        )
        with urlopen(request) as response:
            payload = json.loads(response.read())
        if not isinstance(payload, list):
            raise RuntimeError(f"Supabase returned a non-list payload for {table}")
        return payload

    def get_all(self, table: str, select: str) -> pd.DataFrame:
        rows: list[dict[str, object]] = []
        for offset in range(0, 100_000, TABLE_PAGE_SIZE):
            page = self.get_page(table, select, offset)
            rows.extend(page)
            if len(page) < TABLE_PAGE_SIZE:
                break
        else:
            raise RuntimeError(f"Pagination limit reached for {table}")
        return pd.DataFrame(rows)


def fetch_dataset(client: SupabaseRestClient) -> pd.DataFrame:
    stations = client.get_all(
        "station", "station_id,station_name,corridor,latitude,longitude"
    )
    observations = client.get_all(
        "demand_observation", "station_id,observed_at,demand"
    )
    context = client.get_all(
        "context_observation",
        "observed_at,event_intensity,rain_forecast,rain_mm,temperature_c,temperature_forecast",
    )
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
    if dataset.isna().any().any():
        raise ValueError("Null values found after joining the modeling tables")
    if dataset.duplicated(["station_id", "observed_at"]).any():
        raise ValueError("Duplicate station/timestamp keys found")
    return dataset


def add_features(dataset: pd.DataFrame) -> pd.DataFrame:
    result = dataset.copy()
    minutes = result["observed_at"].dt.hour * 60 + result["observed_at"].dt.minute
    result["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
    result["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
    result["weekday_sin"] = np.sin(2 * np.pi * result["observed_at"].dt.dayofweek / 7)
    result["weekday_cos"] = np.cos(2 * np.pi * result["observed_at"].dt.dayofweek / 7)
    return result


def build_pipeline() -> Pipeline:
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
    preprocessor = ColumnTransformer(
        [
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("numeric", "passthrough", numeric),
        ],
        remainder="drop",
    )
    return Pipeline(
        [("preprocess", preprocessor), ("model", LinearRegression())]
    )


def main() -> None:
    env = load_env()
    url = env.get("SUPABASE_URL")
    key = env.get("SUPABASE_SERVICE_ROLE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY are required")

    dataset = add_features(fetch_dataset(SupabaseRestClient(url, key)))
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
    model = build_pipeline()
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
    predictions = model.predict(test[predictors])
    absolute_error = np.abs(test[target].to_numpy() - predictions)
    wape = float(absolute_error.sum() / test[target].sum())
    feature_names = model.named_steps["preprocess"].get_feature_names_out()
    coefficients = model.named_steps["model"].coef_
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
        "cross_validation": {
            "method": "TimeSeriesSplit(n_splits=5)",
            "mae_mean": float(-cv_scores["test_mae"].mean()),
            "mae_std": float(cv_scores["test_mae"].std()),
            "rmse_mean": float(-cv_scores["test_rmse"].mean()),
            "rmse_std": float(cv_scores["test_rmse"].std()),
            "r2_mean": float(cv_scores["test_r2"].mean()),
            "r2_std": float(cv_scores["test_r2"].std()),
        },
        "test": {
            "mae": float(mean_absolute_error(test[target], predictions)),
            "mse": float(mean_squared_error(test[target], predictions)),
            "rmse": float(np.sqrt(mean_squared_error(test[target], predictions))),
            "r2": float(r2_score(test[target], predictions)),
            "wape": wape,
            "accuracy": 100 * max(0.0, 1.0 - wape),
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
    pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    ).sort_values("coefficient", key=lambda values: values.abs(), ascending=False).to_csv(
        ARTIFACT_DIR / "coefficients.csv", index=False
    )
    pd.DataFrame(
        {
            "observed_at": test["observed_at"].astype(str),
            "station_id": test["station_id"],
            "actual_demand": test[target],
            "predicted_demand": predictions,
        }
    ).to_csv(ARTIFACT_DIR / "predictions.csv", index=False)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved results to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()