from datetime import datetime, timezone

import pandas as pd

from src.accuracy import accuracy_from_wape, forecast_accuracy_frame, wape
from src.pipeline import retrain_decision


NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def model(trained_at: str = "2026-09-30T09:00:00+00:00") -> dict:
    return {"id": 1, "artifact_base64": "abc", "trained_at": trained_at}


def test_retrains_without_active_model() -> None:
    assert retrain_decision(None, None, NOW, 79, 1)[0]
    assert retrain_decision({"id": 1, "trained_at": "2026-09-30T11:59:00+00:00"}, 90.0, NOW, 79, 1)[0]


YOUNG = "2026-09-30T11:30:00+00:00"


def test_keeps_young_model_while_accuracy_is_at_or_above_threshold() -> None:
    assert not retrain_decision(model(YOUNG), 86.8, NOW, 79, 1)[0]
    assert not retrain_decision(model(YOUNG), 79.0, NOW, 79, 1)[0]


def test_retrains_when_accuracy_drops_below_threshold() -> None:
    retrain, reason = retrain_decision(model(), 78.5, NOW, 79, 1)
    assert retrain
    assert "78.50%" in reason


def test_keeps_young_model_when_accuracy_is_unavailable() -> None:
    assert not retrain_decision(model(YOUNG), None, NOW, 79, 1)[0]


def test_retrains_on_schedule_even_with_good_metrics() -> None:
    retrain, reason = retrain_decision(model("2026-09-30T11:00:00+00:00"), 91.9, NOW, 79, 1, wape_ratio=0.59, schedule_hours=1)
    assert retrain
    assert reason.startswith("scheduled retrain")
    assert retrain_decision(model(), None, NOW, 79, 1, schedule_hours=1)[0]
    assert not retrain_decision(model(YOUNG), 91.9, NOW, 79, 1, wape_ratio=0.59, schedule_hours=1)[0]
    assert not retrain_decision(model(), 91.9, NOW, 79, 1, wape_ratio=0.59, schedule_hours=0)[0]


def test_low_accuracy_waits_for_minimum_interval() -> None:
    assert not retrain_decision(model("2026-09-30T11:30:00+00:00"), 70.0, NOW, 79, 1)[0]
    assert retrain_decision(model("2026-09-30T11:00:00"), 70.0, NOW, 79, 1)[0]


def test_accuracy_from_forecasts_matched_with_observed_demand() -> None:
    forecasts = pd.DataFrame({
        "station_id": ["2300", "03000", "03000"],
        "target_at": ["2026-09-10T00:00:00+00:00", "2026-09-10T00:00:00+00:00", "2026-09-10T00:15:00+00:00"],
        "prediction": [90.0, 120.0, 50.0],
    })
    observations = pd.DataFrame({
        "station_id": ["02300", "03000"],
        "observed_at": ["2026-09-10T00:00:00Z", "2026-09-10T00:00:00Z"],
        "demand": [100, 100],
    })
    matched = forecast_accuracy_frame(forecasts, observations)
    assert len(matched) == 2
    value = wape(matched["actual"], matched["prediction"])
    assert value == 0.15
    assert accuracy_from_wape(value) == 85.0


def test_detects_cycle_confirmed_with_other_content() -> None:
    from src.pipeline import submitted_with_other_content

    assert submitted_with_other_content({"status": "confirmed", "payload_sha256": "old"}, "new")
    assert not submitted_with_other_content({"status": "confirmed", "payload_sha256": "same"}, "same")
    assert not submitted_with_other_content({"status": "failed", "payload_sha256": "old"}, "new")
    assert submitted_with_other_content({"status": "failed", "api_submission_id": "sub_1", "payload_sha256": "old"}, "new")
    assert not submitted_with_other_content(None, "new")


def test_only_idempotency_conflicts_count_as_already_submitted() -> None:
    from pulso_transmi.client import PulsoTransmiError
    from src.pipeline import is_idempotency_conflict

    assert is_idempotency_conflict(PulsoTransmiError('HTTP 409: {"code": "idempotency_conflict"}', status_code=409))
    assert not is_idempotency_conflict(PulsoTransmiError("HTTP 409: other conflict", status_code=409))
    assert not is_idempotency_conflict(PulsoTransmiError("HTTP 500: idempotency_conflict", status_code=500))


def test_retrains_when_wape_degrades_even_with_good_accuracy() -> None:
    retrain, reason = retrain_decision(model(), 86.0, NOW, 79, 1, wape_ratio=1.30, ratio_threshold=1.25)
    assert retrain
    assert "WAPE ratio 1.30" in reason
    assert retrain_decision(model(), 86.0, NOW, 79, 1, wape_ratio=1.25, ratio_threshold=1.25)[0]
    assert not retrain_decision(model(YOUNG), 86.0, NOW, 79, 1, wape_ratio=1.10, ratio_threshold=1.25)[0]


def test_wape_trigger_waits_for_minimum_interval() -> None:
    assert not retrain_decision(model("2026-09-30T11:30:00+00:00"), 86.0, NOW, 79, 1, wape_ratio=1.5)[0]
