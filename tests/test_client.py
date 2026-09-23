import hashlib
import json

import httpx

from pulso_transmi import PulsoTransmiClient


def handler(request: httpx.Request) -> httpx.Response:
    if request.url.path == "/v1/submissions":
        assert request.method == "POST"
        assert request.headers["Idempotency-Key"] == "run-123456"
        return httpx.Response(201, json={"id": "sub_test"})
    if request.url.path == "/v1/meta":
        content = b"station_id,name\n03000,Portal Suba\n"
        return httpx.Response(200, json={
            "dataset": {
                "observation_rows": 2,
                "files": {"stations.csv": {"sha256": hashlib.sha256(content).hexdigest()}},
            }
        })
    if request.url.path == "/v1/stations":
        return httpx.Response(200, json={"data": [{"station_id": "03000", "station_name": "Portal Suba"}], "count": 1})
    if request.url.path == "/v1/observations":
        cursor = request.url.params.get("cursor")
        if cursor is None:
            return httpx.Response(200, json={
                "data": [{"observed_at": "2026-07-26T00:00:00-05:00", "station_id": "03000", "demand": 10}],
                "count": 1,
                "next_cursor": "page-2",
            })
        return httpx.Response(200, json={
            "data": [{"observed_at": "2026-07-26T00:15:00-05:00", "station_id": "03000", "demand": 12}],
            "count": 1,
            "next_cursor": None,
        })
    if request.url.path == "/v1/downloads/stations.csv":
        return httpx.Response(200, content=b"station_id,name\n03000,Portal Suba\n")
    return httpx.Response(404, json={"detail": "not found"})


def client() -> PulsoTransmiClient:
    return PulsoTransmiClient(base_url="https://example.test", transport=httpx.MockTransport(handler))


def test_stations_keep_leading_zero() -> None:
    with client() as api:
        stations = api.stations()
    assert stations.iloc[0]["station_id"] == "03000"


def test_all_observation_pages_are_joined() -> None:
    with client() as api:
        observations = api.observations_dataframe()
    assert observations["demand"].tolist() == [10, 12]


def test_download_verifies_checksum(tmp_path) -> None:
    with client() as api:
        path = api.download("stations.csv", tmp_path / "stations.csv")
    assert path.read_text() == "station_id,name\n03000,Portal Suba\n"


def test_submit_sends_idempotency_key() -> None:
    with client() as api:
        response = api.submit({"predictions": []}, idempotency_key="run-123456")
    assert response["id"] == "sub_test"
