"""Drift-robust demand model: recency-weighted, adaptively weighted ensemble.

Lives in ``src`` (not in the training script) because the fitted object is pickled
with joblib and must be importable when the pipeline loads it for inference.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


RANDOM_STATE = 42
CATEGORICAL = ["station_id", "corridor"]
NUMERIC = [
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


def build_preprocessor(*, dense: bool = False) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("categorical", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=not dense)),
            ]), CATEGORICAL),
            ("numeric", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", RobustScaler()),
            ]), NUMERIC),
        ],
        remainder="drop",
    )


def default_members() -> dict[str, Pipeline]:
    """Diverse members: two boosted trees with robust losses and a linear model that extrapolates."""
    boosting = dict(learning_rate=0.05, max_iter=300, max_leaf_nodes=31, min_samples_leaf=30, l2_regularization=1.0, random_state=RANDOM_STATE)
    return {
        # Poisson loss suits non-negative demand counts and multiplicative shifts.
        "hgb_poisson": Pipeline([("preprocess", build_preprocessor(dense=True)), ("model", HistGradientBoostingRegressor(loss="poisson", **boosting))]),
        # Absolute error is not dragged by demand spikes (events, anomalies).
        "hgb_absolute": Pipeline([("preprocess", build_preprocessor(dense=True)), ("model", HistGradientBoostingRegressor(loss="absolute_error", **boosting))]),
        # Trees cannot extrapolate beyond the training range; the linear member can.
        "ridge": Pipeline([("preprocess", build_preprocessor()), ("model", Ridge(alpha=10.0))]),
    }


def recency_weights(timestamps: pd.Series, half_life_days: float, floor: float = 0.05) -> np.ndarray:
    """Exponential decay by age: the newest row weighs 1, a row half_life_days older weighs 0.5."""
    times = pd.to_datetime(timestamps, utc=True)
    age_days = (times.max() - times).dt.total_seconds().to_numpy() / 86400
    return np.maximum(floor, 0.5 ** (age_days / half_life_days))


class DriftRobustEnsemble(BaseEstimator, RegressorMixin):
    """Average of diverse members, weighted by their error on the most recent data.

    - Members are refit on all rows with the caller's (recency) sample weights.
    - Member weights are proportional to 1 / MAE^2 on the chronologically last
      ``holdout_fraction`` of the training rows, so after a strong drift the
      members that cope best with the new regime dominate.
    - Predictions are clipped to [0, max_multiplier x max training demand]; a linear
      member extrapolating far outside the training range cannot explode.

    Rows must be in chronological order (the training script sorts them).
    """

    def __init__(self, holdout_fraction: float = 0.15, max_multiplier: float = 1.5):
        self.holdout_fraction = holdout_fraction
        self.max_multiplier = max_multiplier

    @staticmethod
    def _fit_member(member: Pipeline, X: pd.DataFrame, y: np.ndarray, weights: np.ndarray) -> Pipeline:
        step = member.steps[-1][0]
        return member.fit(X, y, **{f"{step}__sample_weight": weights})

    def fit(self, X: pd.DataFrame, y: Any, sample_weight: np.ndarray | None = None) -> "DriftRobustEnsemble":
        y = np.asarray(y, dtype=float)
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        split = int(len(y) * (1 - self.holdout_fraction))
        holdout_mae: dict[str, float] = {}
        for name, member in default_members().items():
            fitted = self._fit_member(member, X.iloc[:split], y[:split], weights[:split])
            predicted = np.maximum(0.0, fitted.predict(X.iloc[split:]))
            holdout_mae[name] = float(np.mean(np.abs(y[split:] - predicted)))
        inverse = {name: 1.0 / max(mae, 1e-6) ** 2 for name, mae in holdout_mae.items()}
        total = sum(inverse.values())
        self.member_weights_ = {name: value / total for name, value in inverse.items()}
        self.holdout_mae_ = holdout_mae
        self.members_ = {name: self._fit_member(member, X, y, weights) for name, member in default_members().items()}
        self.upper_bound_ = float(np.max(y)) * self.max_multiplier
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        combined = sum(self.member_weights_[name] * member.predict(X) for name, member in self.members_.items())
        return np.clip(combined, 0.0, self.upper_bound_)


class RecencyWeighted(BaseEstimator, RegressorMixin):
    """Wraps an estimator so it always trains with recency weights computed from ``timestamps``.

    The training script passes ``timestamps`` as a fit parameter (sliced per CV fold by
    ``cross_validate``); inference only calls ``predict`` with the usual predictors.
    """

    def __init__(self, estimator: BaseEstimator, half_life_days: float = 14.0):
        self.estimator = estimator
        self.half_life_days = half_life_days

    def fit(self, X: pd.DataFrame, y: Any, timestamps: Any = None) -> "RecencyWeighted":
        weights = None if timestamps is None else recency_weights(pd.Series(np.asarray(timestamps)), self.half_life_days)
        self.estimator_ = clone(self.estimator).fit(X, y, sample_weight=weights)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.estimator_.predict(X)
