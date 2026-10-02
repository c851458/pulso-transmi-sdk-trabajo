import pickle

import numpy as np
import pandas as pd

from src.ingest import combine_observations
from src.lag_model import LagEnsembleForecaster, demand_matrix


STATIONS = ["02300", "03000"]


def observations(days: int = 10, shift_at_day: int | None = None) -> pd.DataFrame:
    """Daily demand profile per station; after ``shift_at_day`` it becomes a 4-hour cycle at a new level."""
    rng = np.random.default_rng(0)
    times = pd.date_range("2026-09-01", periods=days * 96, freq="15min", tz="UTC")
    rows = []
    for offset, station in enumerate(STATIONS):
        minutes = times.hour * 60 + times.minute
        demand = 300 + 100 * offset + 150 * np.sin(2 * np.pi * minutes / 1440)
        if shift_at_day is not None:
            shifted = times >= times[0] + pd.Timedelta(days=shift_at_day)
            demand = np.where(shifted, 900 + 400 * np.sin(2 * np.pi * minutes / 240), demand)
        demand = np.maximum(0, demand + rng.normal(scale=10, size=len(times)))
        rows.append(pd.DataFrame({"station_id": station, "observed_at": times, "demand": demand}))
    return pd.concat(rows, ignore_index=True)


def wape(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.abs(actual - predicted).sum() / actual.sum())


def test_demand_matrix_is_a_regular_grid_with_gaps_as_nan() -> None:
    frame = observations(days=1).drop(index=[3])
    matrix = demand_matrix(frame)
    assert matrix.shape == (96, 2)
    assert matrix.isna().sum().sum() == 1
    assert str(matrix.index.tz) == "UTC"


def test_ensemble_tracks_a_regime_change_that_persistence_misses() -> None:
    history = observations(days=10, shift_at_day=8)
    model = LagEnsembleForecaster().fit(history[history["observed_at"] < "2026-09-09"])
    demand = demand_matrix(history)
    last_day = demand.loc["2026-09-10T12:00Z":]
    ensemble = model.predict_matrix(demand, 4).loc[last_day.index]
    persistence = demand.shift(4).loc[last_day.index]
    assert wape(last_day.to_numpy(), ensemble.to_numpy()) < 0.5 * wape(last_day.to_numpy(), persistence.to_numpy())


def test_forecast_uses_only_past_data_and_covers_every_station_and_target() -> None:
    history = observations(days=9)
    model = LagEnsembleForecaster().fit(history)
    cutoff = history["observed_at"].max()
    targets = [cutoff + pd.Timedelta(minutes=15 * k) for k in range(1, 5)]
    stations = [*STATIONS, "09999"]  # a station without history falls back to the network mean
    forecast = model.forecast(history, targets, stations)
    assert len(forecast) == len(targets) * len(stations)
    assert forecast["value"].notna().all() and (forecast["value"] >= 0).all()


def test_predictions_ignore_demand_at_and_after_the_target() -> None:
    history = observations(days=9)
    model = LagEnsembleForecaster().fit(history)
    demand = demand_matrix(history)
    tampered = demand.copy()
    tampered.iloc[-4:] = 1e6  # the target row and the three after it
    target = demand.index[-4]
    assert np.allclose(model.predict_matrix(tampered, 4).loc[target], model.predict_matrix(demand, 4).loc[target])


def test_fitted_forecaster_survives_pickling_for_the_pipeline() -> None:
    history = observations(days=9)
    model = LagEnsembleForecaster().fit(history)
    targets = [history["observed_at"].max() + pd.Timedelta(minutes=15)]
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(restored.forecast(history, targets, STATIONS)["value"], model.forecast(history, targets, STATIONS)["value"])


def test_stream_observations_are_combined_from_the_latest_demand() -> None:
    starter = pd.DataFrame({"station_id": ["2300", "2300"], "observed_at": pd.to_datetime(["2026-09-09T04:30Z", "2026-09-09T04:45Z"]), "demand": [10, 11]})
    stream = pd.DataFrame({
        "station_id": ["02300", "02300", "02300"],
        "observed_at": pd.to_datetime(["2026-09-09T04:45Z", "2026-09-09T05:00Z", "2026-09-09T03:00Z"]),
        "demand": [12, 13, 1],
        "released_at": "2026-09-21T15:30:00Z",
    })
    combined = combine_observations(starter, stream, pd.Timestamp("2026-09-09T04:30Z"))
    assert list(combined.columns) == ["station_id", "observed_at", "demand"]
    assert combined.sort_values("observed_at")["demand"].tolist() == [10, 12, 13]
    assert set(combined["station_id"]) == {"02300"}
