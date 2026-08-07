# File role: In-memory pub/sub hub that fans live alert messages out to a user's
# WebSocket connections. Upload handlers run on FastAPI's threadpool (sync def)
# while WebSocket coroutines run on the event loop, so publish() is thread-safe
# via loop.call_soon_threadsafe.
# Key symbols/vars: AlertHub, alert_hub.
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

_QUEUE_MAXSIZE = 100


class AlertHub:
    """Per-user alert fan-out.

    subscribe() is called from the WebSocket coroutine (captures the running
    loop). publish() may be called from any thread (the sync sample-upload
    handler); it schedules a put on each subscriber's asyncio.Queue on that
    queue's owning loop.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subs: Dict[str, Set[asyncio.Queue]] = {}
        self._loops: Dict[asyncio.Queue, asyncio.AbstractEventLoop] = {}

    def subscribe(self, user_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=_QUEUE_MAXSIZE)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None  # pragma: no cover - only reachable outside an event loop
        with self._lock:
            self._subs.setdefault(user_id, set()).add(queue)
            if loop is not None:
                self._loops[queue] = loop
        return queue

    def unsubscribe(self, user_id: str, queue: asyncio.Queue) -> None:
        with self._lock:
            subs = self._subs.get(user_id)
            if subs:
                subs.discard(queue)
                if not subs:
                    self._subs.pop(user_id, None)
            self._loops.pop(queue, None)

    def publish(self, user_id: str, payload: dict) -> int:
        """Fan a message out to all of a user's connections.

        Safe to call from a worker thread. Returns the number of connections
        the message was scheduled for (0 when nobody is listening).
        """
        with self._lock:
            queues: List[asyncio.Queue] = list(self._subs.get(user_id, ()))

        if not queues:
            return 0

        def _safe_put(q: asyncio.Queue, msg: dict) -> None:
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                # Slow consumer - drop the alert rather than blocking the
                # producer or buffering unboundedly.
                logger.warning("Dropping live alert for user %s (queue full)", user_id)

        scheduled = 0
        for queue in queues:
            loop = self._loops.get(queue)
            if loop is None:
                continue
            try:
                loop.call_soon_threadsafe(_safe_put, queue, payload)
                scheduled += 1
            except RuntimeError:
                # Loop is shutting down; the WS handler will clean up.
                continue
        return scheduled

    def subscriber_count(self, user_id: str) -> int:
        with self._lock:
            return len(self._subs.get(user_id, ()))


alert_hub = AlertHub()
