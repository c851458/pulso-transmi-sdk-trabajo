import importlib.util
from pathlib import Path


def load_training_module():
    path = Path(__file__).resolve().parents[1] / "examples" / "03_supabase_linear_baseline.py"
    spec = importlib.util.spec_from_file_location("training_baseline", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSupabase:
    """Mimics SupabaseRestClient.get_all, which returns a list of dicts."""

    rows = [
        {"station_id": "1", "observed_at": "2026-09-01T00:15:00+00:00", "demand": 12},
        {"station_id": "00001", "observed_at": "2026-09-01T00:00:00+00:00", "demand": 10},
        {"station_id": "00001", "observed_at": "2026-09-01T00:15:00+00:00", "demand": 12},
    ]

    def get_all(self, table: str, select: str, order: str | None = None):
        assert table == "demand_observation"
        return list(self.rows)


def test_fetch_observations_accepts_list_rows_from_shared_client() -> None:
    observations = load_training_module().fetch_observations(FakeSupabase())

    assert len(observations) == 2
    assert list(observations["demand"]) == [10, 12]
    assert set(observations["station_id"]) == {"00001"}
    assert str(observations["observed_at"].dt.tz) == "UTC"
