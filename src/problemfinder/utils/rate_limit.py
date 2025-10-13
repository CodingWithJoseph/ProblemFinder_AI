"""Concurrency helpers for controlling API throughput."""

from __future__ import annotations

import contextlib
import threading
import time
from typing import Iterator


class RateLimiter:
    """Thread-safe rate limiter expressed in requests per minute."""

    def __init__(self, rate_limit: int) -> None:
        self._rate_limit = max(rate_limit, 0)
        self._lock = threading.Lock()
        self._interval = 60.0 / self._rate_limit if self._rate_limit else 0.0
        self._next_allowed = time.time()

    def acquire(self) -> None:
        """Block until the caller is permitted to proceed."""

        if self._interval <= 0:
            return
        with self._lock:
            now = time.time()
            wait_time = max(0.0, self._next_allowed - now)
            self._next_allowed = max(self._next_allowed, now) + self._interval
        if wait_time:
            time.sleep(wait_time)

    @contextlib.contextmanager
    def slot(self) -> Iterator[None]:
        """Context manager that acquires a rate-limited slot."""

        self.acquire()
        yield
