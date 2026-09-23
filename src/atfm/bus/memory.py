from __future__ import annotations

from collections import deque
from threading import Lock

from atfm.schema.events import Event


class InMemoryBus:
    def __init__(self):
        self._q: deque = deque()
        self._lock = Lock()

    def publish(self, e: Event) -> None:
        with self._lock:
            self._q.append(e)

    def drain(self) -> list[Event]:
        with self._lock:
            out = list(self._q)
            self._q.clear()
        return out
