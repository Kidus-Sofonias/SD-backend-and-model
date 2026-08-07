"""Lightweight in-memory rate limiting for abuse-prone endpoints.

Simple per-key sliding-window counters guarded by a lock. State is per-process,
which is acceptable for single-worker deployments (Render free/standard tier)
and local development. Multi-worker deployments should back this with a shared
store (e.g. Redis) — see docs/CODEBASE_REVIEW_PHASE1.md H-7.
"""

from __future__ import annotations

import threading
import time
from collections import deque

from fastapi import Request


class SlidingWindowRateLimiter:
    # Upper bound on tracked keys to prevent unbounded memory growth under a
    # distributed attack (e.g. rotating spoofed X-Forwarded-For headers). The
    # dict is insertion-ordered, so evicting the oldest key is O(1).
    MAX_KEYS = 100_000

    def __init__(self, *, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            if key not in self._hits:
                if len(self._hits) >= self.MAX_KEYS:
                    self._hits.pop(next(iter(self._hits)))
                self._hits[key] = deque()
            window = self._hits[key]
            while window and now - window[0] > self.window_seconds:
                window.popleft()
            if len(window) >= self.max_requests:
                return False
            window.append(now)
            return True


# Auth: 10 login/register attempts per minute per client IP.
LOGIN_RATE_LIMITER = SlidingWindowRateLimiter(max_requests=10, window_seconds=60.0)
# Sensor uploads: 120 batches per minute per user. The mobile app uploads a batch
# roughly every 4 seconds (~15/min), so this leaves generous headroom while
# stopping runaway loops or abuse.
UPLOAD_RATE_LIMITER = SlidingWindowRateLimiter(max_requests=120, window_seconds=60.0)


def client_ip_for(request: Request) -> str:
    """Best-effort client IP, honoring a single proxy hop via X-Forwarded-For.

    Note: X-Forwarded-For is client-spoofable, so this is a convenience key for
    the login rate limiter, not an authentication boundary. Behind a trusted
    reverse proxy (Render) it reflects the real client reasonably well.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip() or "unknown"
    if request.client is not None:
        return str(request.client.host)
    return "unknown"
