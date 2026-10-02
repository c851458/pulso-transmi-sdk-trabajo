"""Short-horizon demand forecaster that adapts to regime changes without retraining.

Cycles ask for 1-4 periods (15-60 min) ahead, and the competition stream switches
regimes without warning: station levels jump and the daily profile can turn into a
4-hour cycle. The forecaster combines simple lag forecasters (persistence and
4-hour, daily and weekly seasonal naives) with a gradient-boosted correction of
persistence, and weights them per station and target by their error over the last
``error_window`` periods, so whichever member tracks the current regime dominates
within a few hours.

Lives in ``src`` (not in the training script) because the fitted object is pickled
with joblib and must be importable when the pipeline loads it for inference.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import HistGradientBoostingRegressor

from src.robust_model import recency_weights


FREQ = "15min"
PERIOD = pd.Timedelta(FREQ)
TIMEZONE = "America/Bogota"
HORIZONS = (1, 2, 3, 4)
# Seasonal lags in periods: 4 hours, 1 day, 1 week.
SEASONS = (16, 96, 672)
# Observations needed before the first target: the weekly lag, its horizon offset and the error window.
HISTORY_PERIODS = max(SEASONS) + max(HORIZONS) + 64
RANDOM_STATE = 42


def feature_names(seasons: tuple[int, ...]) -> list[str]:
    return [
        "persistence",
        "mean_1h",
        "mean_4h",
        *[f"seasonal_{season}" for season in seasons],
        *[f"seasonal_{season}_delta" for season in seasons],
        "hour_sin",
        "hour_cos",
        "weekday",
        "horizon",
        "station_code",
    ]


def demand_matrix(observations: pd.DataFrame) -> pd.DataFrame:
    """Observations (station_id, observed_at, demand) as a regular 15-min UTC time x station grid."""
    frame = observations.assign(
        station_id=observations["station_id"].astype(str).str.zfill(5),
        observed_at=pd.to_datetime(observations["observed_at"], utc=True),
        demand=pd.to_numeric(observations["demand"], errors="coerce"),
    )
    wide = (
        frame.drop_duplicates(["station_id", "observed_at"], keep="last")
        .pivot(index="observed_at", columns="station_id", values="demand")
        .sort_index()
    )
    return wide.asfreq(FREQ).astype(float)


def lag_members(demand: pd.DataFrame, horizon: int, seasons: tuple[int, ...] = SEASONS) -> dict[str, pd.DataFrame]:
    """Each member's forecast for every row (target time), using only data known ``horizon`` periods earlier."""
    last = demand.shift(horizon)
    members = {"persistence": last, "mean_1h": last.rolling(4).mean()}
    for season in seasons:
        members[f"seasonal_{season}"] = demand.shift(season)
        # Latest value plus the change the series had over the same span one season ago.
        members[f"seasonal_{season}_delta"] = last + demand.shift(season) - demand.shift(season + horizon)
    return members


class LagEnsembleForecaster(BaseEstimator, RegressorMixin):
    """Lag forecasters plus a boosted persistence correction, weighted by recent error.

    Member weight = 1 / (recent MAE ** weight_power + 1), computed per station and
    target from the ``error_window`` periods that end at the forecast cutoff.
    """

    # Class-level defaults keep artifacts pickled before these parameters existed loadable.
    seasons: tuple[int, ...] = SEASONS
    train_horizons: tuple[int, ...] = HORIZONS

    def __init__(
        self,
        half_life_days: float = 7.0,
        error_window: int = 16,
        weight_power: float = 4.0,
        seasons: tuple[int, ...] = SEASONS,
        train_horizons: tuple[int, ...] = HORIZONS,
    ):
        self.half_life_days = half_life_days
        self.error_window = error_window
        self.weight_power = weight_power
        self.seasons = seasons
        self.train_horizons = train_horizons

    def _features(self, demand: pd.DataFrame, horizon: int) -> pd.DataFrame:
        columns = {name: frame.stack(future_stack=True) for name, frame in lag_members(demand, horizon, self.seasons).items()}
        columns["mean_4h"] = demand.shift(horizon).rolling(16).mean().stack(future_stack=True)
        features = pd.DataFrame(columns)
        features.index.names = ["observed_at", "station_id"]
        local = features.index.get_level_values("observed_at").tz_convert(TIMEZONE)
        minutes = local.hour * 60 + local.minute
        features["hour_sin"] = np.sin(2 * np.pi * minutes / 1440)
        features["hour_cos"] = np.cos(2 * np.pi * minutes / 1440)
        features["weekday"] = local.dayofweek
        features["horizon"] = horizon
        codes = {station: code for code, station in enumerate(self.stations_)}
        features["station_code"] = features.index.get_level_values("station_id").map(lambda station: codes.get(station, -1))
        return features

    def fit(self, observations: pd.DataFrame, y: object = None) -> "LagEnsembleForecaster":
        demand = demand_matrix(observations)
        self.stations_ = list(demand.columns)
        frames = []
        for horizon in self.train_horizons:
            features = self._features(demand, horizon)
            features["target"] = demand.stack(future_stack=True).reindex(features.index)
            frames.append(features.dropna(subset=["target", "persistence"]))
        train = pd.concat(frames)
        weights = recency_weights(pd.Series(train.index.get_level_values("observed_at")), self.half_life_days)
        self.correction_ = HistGradientBoostingRegressor(
            loss="absolute_error",
            learning_rate=0.05,
            max_iter=300,
            min_samples_leaf=40,
            random_state=RANDOM_STATE,
        ).fit(train[feature_names(self.seasons)], train["target"] - train["persistence"], sample_weight=weights)
        self.training_end_ = demand.index.max()
        return self

    def member_forecasts(self, demand: pd.DataFrame, horizon: int) -> dict[str, pd.DataFrame]:
        members = lag_members(demand, horizon, self.seasons)
        features = self._features(demand, horizon).dropna(subset=["persistence"])
        corrected = pd.Series(np.nan, index=features.index)
        if not features.empty:
            corrected = features["persistence"] + self.correction_.predict(features[feature_names(self.seasons)])
        members["boosted_correction"] = corrected.unstack().reindex(index=demand.index, columns=demand.columns)
        return members

    def predict_matrix(self, demand: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Ensemble forecast for every row of ``demand`` made ``horizon`` periods before it."""
        members = self.member_forecasts(demand, horizon)
        numerator = pd.DataFrame(0.0, index=demand.index, columns=demand.columns)
        denominator = numerator.copy()
        available_sum = numerator.copy()
        available_count = numerator.copy()
        for forecast in members.values():
            error = (demand - forecast).abs().rolling(self.error_window, min_periods=self.error_window // 2).mean().shift(horizon)
            weight = (1.0 / (error**self.weight_power + 1.0)).where(forecast.notna() & error.notna(), 0.0)
            numerator += forecast.fillna(0.0) * weight
            denominator += weight
            available_sum += forecast.fillna(0.0)
            available_count += forecast.notna()
        # Without recent errors to rank members (sparse history), fall back to their plain mean.
        combined = (numerator / denominator).where(denominator > 0, available_sum / available_count.replace(0, np.nan))
        return combined.clip(lower=0.0)

    def forecast(self, observations: pd.DataFrame, targets: Iterable[pd.Timestamp], stations: Iterable[str]) -> pd.DataFrame:
        """Forecast each station at each target time from the observations available now."""
        demand = demand_matrix(observations)
        stations = [str(station).zfill(5) for station in stations]
        targets = sorted({pd.Timestamp(target).tz_convert("UTC") for target in targets})
        latest = demand.dropna(how="all").index.max()
        grid = pd.date_range(demand.index.min(), max(max(targets), latest), freq=FREQ)
        demand = demand.reindex(index=grid, columns=sorted(set(demand.columns) | set(stations)))
        # Unforecastable cells (station without history) fall back to its recent mean, then the network mean.
        recent = demand.loc[:latest].tail(96)
        fallback = recent.mean().fillna(float(np.nanmean(recent.to_numpy())) if recent.notna().any().any() else 0.0)
        rows = []
        for horizon, group in pd.Series(targets).groupby(lambda i: max(1, round((targets[i] - latest) / PERIOD))):
            predicted = self.predict_matrix(demand, int(horizon))
            for target in group:
                values = predicted.loc[target, stations].fillna(fallback[stations])
                rows.extend({"station_id": station, "target_at": target, "value": float(value)} for station, value in values.items())
        return pd.DataFrame(rows, columns=["station_id", "target_at", "value"])
