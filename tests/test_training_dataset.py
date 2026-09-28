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

    tables = {
        "station": [{"station_id": "00001", "station_name": "A", "corridor": "C", "latitude": 4.6, "longitude": -74.1}],
        "demand_observation": [
            {"station_id": "00001", "observed_at": "2026-09-01T00:00:00+00:00", "demand": 10},
            {"station_id": "00001", "observed_at": "2026-09-01T00:15:00+00:00", "demand": 12},
        ],
        "context_observation": [
            {"observed_at": ts, "event_intensity": 0, "rain_forecast": 0, "rain_mm": 0, "temperature_c": 14, "temperature_forecast": 14}
            for ts in ("2026-09-01T00:00:00+00:00", "2026-09-01T00:15:00+00:00")
        ],
    }

    def get_all(self, table: str, select: str):
        return list(self.tables[table])


def test_fetch_dataset_accepts_list_rows_from_shared_client() -> None:
    dataset = load_training_module().fetch_dataset(FakeSupabase())

    assert len(dataset) == 2
    assert list(dataset["demand"]) == [10, 12]
    assert str(dataset["observed_at"].dt.tz) == "UTC"
