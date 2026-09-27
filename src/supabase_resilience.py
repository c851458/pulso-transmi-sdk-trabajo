"""Small, bounded retry policy for read-only Supabase REST requests."""

from __future__ import annotations

import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


TRANSIENT_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504, 522}
MAX_GET_ATTEMPTS = 3


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
