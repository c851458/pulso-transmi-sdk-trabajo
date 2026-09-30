import pickle

import numpy as np
import pandas as pd

from src.robust_model import CATEGORICAL, NUMERIC, DriftRobustEnsemble, RecencyWeighted, recency_weights


def dataset(rows: int = 600, shift_at: int | None = None) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    rng = np.random.default_rng(0)
    times = pd.Series(pd.date_range("2026-08-01", periods=rows, freq="h", tz="UTC"))
    X = pd.DataFrame({name: rng.normal(size=rows) for name in NUMERIC})
    X["station_id"] = rng.choice(["02300", "03000"], size=rows)
    X["corridor"] = "Caracas"
    X = X[CATEGORICAL + NUMERIC]
    y = 200 + 40 * X["hour_sin"].to_numpy() + rng.normal(scale=5, size=rows)
    if shift_at is not None:
        y[shift_at:] += 150  # strong regime change at the end of the history
    return X, np.maximum(y, 0), times


def test_recency_weights_halve_every_half_life() -> None:
    times = pd.Series(pd.to_datetime(["2026-09-01", "2026-09-15", "2026-09-29"], utc=True))
    weights = recency_weights(times, half_life_days=14, floor=0.0)
    assert np.allclose(weights, [0.25, 0.5, 1.0])
    assert recency_weights(times, half_life_days=1).min() == 0.05


def test_ensemble_weights_members_by_recent_error_and_clips_predictions() -> None:
    X, y, _ = dataset()
    model = DriftRobustEnsemble().fit(X, y)
    assert set(model.member_weights_) == {"hgb_poisson", "hgb_absolute", "ridge"}
    assert abs(sum(model.member_weights_.values()) - 1) < 1e-9
    extreme = X.head(5).assign(event_intensity=1e6, temperature_c=-1e6)
    predictions = model.predict(extreme)
    assert (predictions >= 0).all() and (predictions <= model.upper_bound_).all()


def test_recency_weighting_adapts_faster_to_a_regime_change() -> None:
    X, y, times = dataset(shift_at=450)
    recent = X.tail(50)
    weighted = RecencyWeighted(DriftRobustEnsemble(), half_life_days=2).fit(X, y, timestamps=times.to_numpy())
    unweighted = DriftRobustEnsemble().fit(X, y)
    error_weighted = np.abs(weighted.predict(recent) - y[-50:]).mean()
    error_unweighted = np.abs(unweighted.predict(recent) - y[-50:]).mean()
    assert error_weighted < error_unweighted


def test_fitted_model_survives_pickling_for_the_pipeline() -> None:
    X, y, times = dataset()
    model = RecencyWeighted(DriftRobustEnsemble()).fit(X, y, timestamps=times.to_numpy())
    restored = pickle.loads(pickle.dumps(model))
    assert np.allclose(restored.predict(X.head(10)), model.predict(X.head(10)))
