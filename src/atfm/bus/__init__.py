from typing import Protocol

from atfm.schema.events import Event

from .jsonl import JsonlBus, read_events
from .memory import InMemoryBus


class Bus(Protocol):
    def publish(self, e: Event) -> None: ...

    def drain(self) -> list[Event]: ...


__all__ = ["Bus", "InMemoryBus", "JsonlBus", "read_events"]
