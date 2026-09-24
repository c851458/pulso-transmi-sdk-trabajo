from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import time
import uuid
from pathlib import Path
from typing import Any, Iterator

import httpx
import pandas as pd


DEFAULT_BASE_URL = "https://pulso-transmi.72-60-245-2.sslip.io"
LOG = logging.getLogger("pulso.api")


class PulsoTransmiError(RuntimeError):
    """Raised when the Pulso TransMi API cannot fulfill a request."""

    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


class PulsoTransmiClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = 3,
        backoff_seconds: tuple[float, ...] = (5.0, 15.0, 30.0),
    ) -> None:
        resolved_url = base_url or os.getenv("PULSO_API_URL", DEFAULT_BASE_URL)
        resolved_key = api_key or os.getenv("PULSO_API_KEY")
        self._secret = resolved_key or ""
        headers = {"User-Agent": "pulso-transmi-python/0.1.0"}
        if resolved_key:
            headers["Authorization"] = f"Bearer {resolved_key}"
        self._client = httpx.Client(
            base_url=resolved_url.rstrip("/"),
            headers=headers,
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if len(backoff_seconds) < max_retries:
            raise ValueError("backoff_seconds must contain one value per retry")
        self._max_retries = max_retries
        self._backoff_seconds = backoff_seconds

    def __enter__(self) -> "PulsoTransmiClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def _response_detail(self, response: httpx.Response) -> str:
        try:
            detail: Any = response.json()
            rendered = json.dumps(detail, ensure_ascii=False, default=str)
        except (ValueError, json.JSONDecodeError):
            rendered = response.text
        if self._secret:
            rendered = rendered.replace(self._secret, "[REDACTED]")
        return rendered[:500].replace("\n", " ")

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        allowed_statuses = kwargs.pop("_allowed_statuses", set())
        for attempt in range(self._max_retries + 1):
            try:
                response = self._client.request(method, path, **kwargs)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError) as exc:
                if attempt >= self._max_retries:
                    raise PulsoTransmiError(
                        f"{method} {path} failed after {attempt + 1} attempts: {type(exc).__name__}",
                        retryable=True,
                    ) from exc
                self._sleep_before_retry(attempt, None)
                continue
            except httpx.HTTPError as exc:
                raise PulsoTransmiError(f"{method} {path} failed: {type(exc).__name__}") from exc

            LOG.info(
                "API response: method=%s path=%s attempt=%d/%d status=%d",
                method,
                path,
                attempt + 1,
                self._max_retries + 1,
                response.status_code,
            )
            retryable = response.status_code == 408 or response.status_code == 429 or response.status_code >= 500
            if response.is_success or response.status_code in allowed_statuses:
                return response
            if retryable and attempt < self._max_retries:
                self._sleep_before_retry(attempt, response)
                continue
            raise PulsoTransmiError(
                f"{method} {path} failed with HTTP {response.status_code}: {self._response_detail(response)}",
                status_code=response.status_code,
                retryable=retryable,
            )
        raise AssertionError("unreachable")

    def _sleep_before_retry(self, attempt: int, response: httpx.Response | None) -> None:
        retry_after = response.headers.get("Retry-After") if response else None
        try:
            delay = float(retry_after) if retry_after is not None else self._backoff_seconds[attempt]
        except ValueError:
            delay = self._backoff_seconds[attempt]
        # Small jitter prevents a group of scheduled jobs retrying together.
        time.sleep(max(0.0, delay + random.uniform(0.0, min(1.0, delay * 0.1))))

    def _get(self, path: str, *, params: dict[str, Any] | None = None) -> httpx.Response:
        return self._request("GET", path, params=params)

    def submit(self, payload: dict[str, Any], *, idempotency_key: str | None = None) -> dict[str, Any]:
        key = idempotency_key or str(uuid.uuid4())
        if not isinstance(payload, dict) or not payload:
            raise PulsoTransmiError("POST /v1/submissions rejected: payload must be a non-empty object")
        response = self._request(
            "POST",
            "/v1/submissions",
            json=payload,
            headers={"Idempotency-Key": key},
        )
        try:
            result = response.json()
        except ValueError as exc:
            raise PulsoTransmiError("POST /v1/submissions returned non-JSON success response", status_code=response.status_code) from exc
        if not isinstance(result, dict) or not result:
            raise PulsoTransmiError("POST /v1/submissions returned an empty confirmation", status_code=response.status_code)
        has_identifier = any(result.get(key) for key in ("id", "submission_id", "operation_id"))
        has_boolean_confirmation = result.get("received") is True or result.get("accepted") is True
        if not (has_identifier or has_boolean_confirmation):
            raise PulsoTransmiError("POST /v1/submissions response lacks a confirmation field", status_code=response.status_code)
        return result

    def current_cycle(self) -> dict[str, Any] | None:
        try:
            # A missing open cycle is a documented business response, not a transport failure.
            response = self._request("GET", "/v1/forecast-cycles/current", _allowed_statuses={404})
            if response.status_code == 404:
                detail = response.json().get("detail", {})
                if isinstance(detail, dict) and detail.get("code") == "no_open_cycle":
                    return None
                raise PulsoTransmiError(
                    f"GET /v1/forecast-cycles/current failed with HTTP 404: {self._response_detail(response)}",
                    status_code=404,
                )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as exc:
            raise PulsoTransmiError(f"GET /v1/forecast-cycles/current failed: {exc}") from exc

    def meta(self) -> dict[str, Any]:
        return self._get("/v1/meta").json()

    def stations(self) -> pd.DataFrame:
        payload = self._get("/v1/stations").json()
        frame = pd.DataFrame(payload["data"])
        if not frame.empty:
            frame["station_id"] = frame["station_id"].astype("string")
        return frame

    def observations_page(
        self,
        *,
        station_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        cursor: str | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        params = {
            "station_id": station_id,
            "start": start,
            "end": end,
            "cursor": cursor,
            "limit": limit,
        }
        return self._get("/v1/observations", params={key: value for key, value in params.items() if value is not None}).json()

    def context_page(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        cursor: str | None = None,
        limit: int = 1000,
    ) -> dict[str, Any]:
        params = {"start": start, "end": end, "cursor": cursor, "limit": limit}
        return self._get("/v1/context", params={key: value for key, value in params.items() if value is not None}).json()

    def _all_pages(self, endpoint: str, params: dict[str, Any]) -> Iterator[dict[str, Any]]:
        cursor = None
        seen: set[str] = set()
        while True:
            page_params = {**params, "cursor": cursor}
            payload = self._get(endpoint, params={key: value for key, value in page_params.items() if value is not None}).json()
            yield from payload["data"]
            cursor = payload.get("next_cursor")
            if cursor is None:
                break
            if cursor in seen:
                raise PulsoTransmiError("API returned a repeated cursor")
            seen.add(cursor)

    def observations_dataframe(
        self,
        *,
        station_id: str | None = None,
        start: str | None = None,
        end: str | None = None,
        page_size: int = 5000,
    ) -> pd.DataFrame:
        rows = self._all_pages(
            "/v1/observations",
            {"station_id": station_id, "start": start, "end": end, "limit": page_size},
        )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
            frame["station_id"] = frame["station_id"].astype("string")
        return frame

    def context_dataframe(
        self,
        *,
        start: str | None = None,
        end: str | None = None,
        page_size: int = 5000,
    ) -> pd.DataFrame:
        rows = self._all_pages(
            "/v1/context", {"start": start, "end": end, "limit": page_size}
        )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame["observed_at"] = pd.to_datetime(frame["observed_at"], utc=True)
        return frame

    def download(self, filename: str, destination: str | Path) -> Path:
        allowed = {"stations.csv", "observations.csv", "context.csv", "metadata.json"}
        if filename not in allowed:
            raise ValueError(f"unsupported filename: {filename}")
        response = self._get(f"/v1/downloads/{filename}")
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)

        if filename != "metadata.json":
            expected = self.meta()["dataset"]["files"][filename]["sha256"]
            actual = hashlib.sha256(response.content).hexdigest()
            if actual != expected:
                path.unlink(missing_ok=True)
                raise PulsoTransmiError(f"checksum mismatch for {filename}")
        return path
