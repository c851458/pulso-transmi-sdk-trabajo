from datetime import datetime, timezone

from src.pipeline import retrain_due


NOW = datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc)


def model(trained_at: str) -> dict:
    return {"id": 1, "artifact_base64": "abc", "trained_at": trained_at}


def test_retrains_without_active_model() -> None:
    assert retrain_due(None, NOW, 1)
    assert retrain_due({"id": 1, "trained_at": "2026-09-30T11:59:00+00:00"}, NOW, 1)


def test_skips_retraining_when_model_is_younger_than_interval() -> None:
    assert not retrain_due(model("2026-09-30T11:10:00+00:00"), NOW, 1)


def test_retrains_once_interval_has_elapsed() -> None:
    assert retrain_due(model("2026-09-30T11:00:00+00:00"), NOW, 1)
    assert retrain_due(model("2026-09-30T09:00:00"), NOW, 1)
