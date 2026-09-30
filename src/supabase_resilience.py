"""Shared Supabase REST client with bounded retries and connection pooling."""

from __future__ import annotations

import json
import time
from typing import Any
from urllib.parse import urlencode
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import httpx


TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
MAX_GET_ATTEMPTS = 3
DEFAULT_TIMEOUT = 30.0


class SupabaseRestClient:
    """One pooled HTTP client for all Supabase calls in a process.

    Supabase REST is HTTP, so keeping a small pool of persistent connections
    avoids opening a new TCP/TLS connection for every table operation. Writes
    are never retried automatically because they are not universally safe to
    repeat; only idempotent GET requests receive bounded retries.
    """

    def __init__(
        self,
        url: str,
        key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = url.rstrip("/") + "/rest/v1"
        self._client = httpx.Client(
            headers={
                "apikey": key,
                "Authorization": f"Bearer {key}",
                "Accept": "application/json",
                "Content-Type": "application/json",
                "Prefer": "return=representation",
            },
            timeout=timeout,
            transport=transport,
            limits=httpx.Limits(max_connections=4, max_keepalive_connections=2, keepalive_expiry=30.0),
            follow_redirects=True,
        )

    def __enter__(self) -> "SupabaseRestClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self._client.close()

    def request(
        self,
        method: str,
        table: str,
        payload: Any = None,
        *,
        params: dict[str, str] | None = None,
        headers: dict[str, str] | None = None,
    ) -> list[dict[str, Any]]:
        method = method.upper()
        attempts = MAX_GET_ATTEMPTS if method == "GET" else 1
        url = f"{self.base_url}/{table}"
        for attempt in range(attempts):
            try:
                response = self._client.request(
                    method,
                    url,
                    params=params,
                    content=None if payload is None else json.dumps(payload, default=str),
                    headers=headers,
                )
                if response.status_code in TRANSIENT_STATUS_CODES and attempt < attempts - 1:
                    response.close()
                    time.sleep(2**attempt)
                    continue
                response.raise_for_status()
                result = response.json()
                response.close()
            except (httpx.TimeoutException, httpx.NetworkError, httpx.ProtocolError) as exc:
                if attempt >= attempts - 1:
                    raise RuntimeError(f"Supabase {method} {table} failed after {attempt + 1} attempts: {type(exc).__name__}") from exc
                time.sleep(2**attempt)
                continue
            except httpx.HTTPStatusError as exc:
                raise RuntimeError(f"Supabase {method} {table} failed with HTTP {exc.response.status_code}: {exc.response.text[:300]}") from exc
            except (ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Supabase returned invalid JSON for {table}") from exc
            if not isinstance(result, list):
                raise RuntimeError(f"Supabase returned an unexpected payload for {table}")
            return result
        raise AssertionError("unreachable")

    def rows(self, table: str, **params: str) -> list[dict[str, Any]]:
        return self.request("GET", table, params=params)

    def get_all(self, table: str, select: str, *, page_size: int = 1_000) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for offset in range(0, 100_000, page_size):
            page = self.rows(table, select=select, limit=str(page_size), offset=str(offset))
            rows.extend(page)
            if len(page) < page_size:
                return rows
        raise RuntimeError(f"Pagination limit reached for {table}")

    def insert(self, table: str, payload: dict[str, Any]) -> dict[str, Any]:
        rows = self.request("POST", table, payload)
        if not rows:
            raise RuntimeError(f"Supabase did not return the inserted {table}")
        return rows[0]

    def insert_many(self, table: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return self.request("POST", table, rows) if rows else []

    def upsert_many(self, table: str, rows: list[dict[str, Any]], conflict_columns: str) -> list[dict[str, Any]]:
        if not rows:
            return []
        return self.request(
            "POST", table, rows,
            params={"on_conflict": conflict_columns},
            headers={"Prefer": "return=representation,resolution=merge-duplicates"},
        )

    def update(self, table: str, filters: dict[str, str], payload: dict[str, Any]) -> list[dict[str, Any]]:
        return self.request("PATCH", table, payload, params=filters)

    def prune_model_history(self, keep_model_id: int) -> dict[str, Any]:
        rows = self.request("POST", "rpc/prune_model_history", {"keep_model_id": keep_model_id})
        return rows[0] if rows else {}

    def latest_data_cut_id(self) -> int:
        rows = self.rows("data_cut", select="id", order="queried_at.desc", limit="1")
        if not rows:
            raise RuntimeError("No data_cut exists in Supabase")
        return int(rows[0]["id"])

    def find_model(self, version: str) -> dict[str, Any] | None:
        rows = self.rows("model", select="id,version,trained_at", version=f"eq.{version}", limit="1")
        return rows[0] if rows else None

    def active_model(self) -> dict[str, Any] | None:
        rows = self.rows(
            "model",
            select="id,version,algorithm,trained_at,artifact_base64,artifact_sha256",
            status="eq.active",
            order="trained_at.desc",
            limit="1",
        )
        return rows[0] if rows else None

    def latest_training_metrics(self, model_id: int) -> dict[str, Any]:
        rows = self.rows("training_run", select="metrics", model_id=f"eq.{model_id}", order="started_at.desc", limit="1")
        return rows[0].get("metrics") or {} if rows else {}


def open_supabase(request: Request, *, timeout: float = 30):
    """Open a Supabase request, retrying only idempotent GETs at most twice."""
    attempts = MAX_GET_ATTEMPTS if (request.method or "GET").upper() == "GET" else 1
    for attempt in range(attempts):
        try:
            return urlopen(request, timeout=timeout)
        except HTTPError as error:
            if error.code not in TRANSIENT_STATUS_CODES or attempt == attempts - 1:
                raise
        except (URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
        time.sleep(2**attempt)

    raise AssertionError("unreachable")
