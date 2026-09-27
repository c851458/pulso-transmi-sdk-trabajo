from datetime import datetime, timezone

from src.drift import execution_error_payload, execution_success_payload


def timestamps() -> tuple[datetime, datetime]:
    return (
        datetime(2026, 9, 25, 12, 0, tzinfo=timezone.utc),
        datetime(2026, 9, 25, 12, 1, tzinfo=timezone.utc),
    )


def test_success_payload_stores_complete_result_and_maximum_psi() -> None:
    started_at, completed_at = timestamps()
    report = {
        "model_id": 7,
        "model_version": "v7",
        "reference_window": {"rows": 100},
        "current_window": {"rows": 110},
        "thresholds": {"psi": 0.2},
        "feature_psi": {"demand": 0.1, "rain_mm": 0.8, "temperature_c": 0.3},
        "drifted_features": ["rain_mm"],
        "performance": {"ratio": 1.1},
        "drift_alert": True,
    }

    payload = execution_success_payload("run-1", started_at, completed_at, report)

    assert payload["execution_id"] == "run-1"
    assert payload["status"] == "success"
    assert payload["drift_score"] == 0.8
    assert payload["drift_detected"] is True
    assert payload["analyzed_features"] == ["demand", "rain_mm", "temperature_c"]
    assert payload["result"] == report


def test_error_payload_records_failure_without_fake_result() -> None:
    started_at, completed_at = timestamps()
    payload = execution_error_payload(
        "run-2", started_at, completed_at, RuntimeError("temporary failure")
    )

    assert payload["status"] == "error"
    assert payload["result"] == {}
    assert payload["error_message"] == "RuntimeError: temporary failure"
