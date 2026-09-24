from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from atfm.schema.events import Event, event_to_dict, parse_event


class JsonlBus:
    """Append-only event log; publish never raises (write failures are counted)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self.errors = 0

    def publish(self, e: Event) -> None:
        try:
            line = json.dumps(event_to_dict(e), separators=(",", ":"))
            with self._lock, open(self.path, "a") as f:
                f.write(line + "\n")
        except Exception:
            self.errors += 1

    def drain(self) -> list[Event]:
        return read_events(self.path)


def read_events(path: str | Path) -> list[Event]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(parse_event(json.loads(line)))
    return out
