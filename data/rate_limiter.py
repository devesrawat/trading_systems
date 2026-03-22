"""
Thread-safe rate limiter for external API calls.

Provides a leaky-bucket implementation that can be used as a context
manager.  Extracted from ``data/bulk_ingest.py`` so it is reusable
across all modules that call rate-limited APIs (Kite, Finnhub, etc.).

Usage::

    limiter = RateLimiter(rps=3)  # max 3 calls/second

    with limiter:
        response = kite.historical_data(...)
"""
from __future__ import annotations

import time
from threading import Semaphore, Thread
from typing import Any


class RateLimiter:
    """
    Leaky-bucket rate limiter: allows at most *rps* calls per second.

    Thread-safe.  Use as a context manager to acquire a slot before
    making a rate-limited API call::

        limiter = RateLimiter(rps=3)
        with limiter:
            resp = api_client.call(...)
    """

    def __init__(self, rps: int) -> None:
        if rps <= 0:
            raise ValueError(f"rps must be > 0, got {rps}")
        self._sem = Semaphore(rps)
        self._rps = rps

    def __enter__(self) -> "RateLimiter":
        self._sem.acquire()
        return self

    def __exit__(self, *_: Any) -> None:
        # Release the acquired slot after 1/rps seconds so the
        # effective rate stays at rps calls per second.
        def _delayed_release(sem: Semaphore, delay: float) -> None:
            time.sleep(delay)
            sem.release()

        Thread(
            target=_delayed_release,
            args=(self._sem, 1.0 / self._rps),
            daemon=True,
        ).start()
