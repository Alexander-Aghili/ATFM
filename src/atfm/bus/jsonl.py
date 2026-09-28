from __future__ import annotations

import json
import os
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
        self.malformed = 0
        self._identity: tuple[int, int] | None = None
        self._offset = 0

    def publish(self, e: Event) -> None:
        try:
            line = json.dumps(event_to_dict(e), separators=(",", ":"))
            with self._lock, open(self.path, "a") as f:
                f.write(line + "\n")
        except Exception:
            self.errors += 1

    def drain(self) -> list[Event]:
        """Consume complete appended lines once per reader lifetime.

        A new reader replays from zero. Replacement or observed truncation resets
        the cursor; copy-truncate followed by regrowth between polls is unsupported.
        Consumption is not an acknowledgement of downstream application or a
        durable checkpoint. I/O failures leave the cursor unchanged for retry.
        """
        with self._lock:
            try:
                stream = self.path.open("rb")
            except FileNotFoundError:
                return []
            with stream:
                return self._read_appended(stream)

    def _read_appended(self, stream):
        stat = os.fstat(stream.fileno())
        identity = (stat.st_dev, stat.st_ino)
        offset = self._offset if identity == self._identity and stat.st_size >= self._offset else 0
        stream.seek(offset)
        events, malformed, offset = self._complete_lines(stream, stat, offset)
        self._identity, self._offset = identity, offset
        self.malformed += malformed
        return events

    def _complete_lines(self, stream, stat, offset):
        events = []
        malformed = 0
        while stream.tell() < stat.st_size:
            line = stream.readline(stat.st_size - stream.tell())
            if not line.endswith(b"\n"):
                break
            offset = stream.tell()
            if not line.strip():
                continue
            try:
                events.append(parse_event(json.loads(line)))
            except (ValueError, UnicodeDecodeError):
                malformed += 1
        return events, malformed, offset


def read_events(path: str | Path) -> list[Event]:
    p = Path(path)
    if not p.exists():
        return []
    out = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(parse_event(json.loads(line)))
            except Exception:
                continue                          # a malformed line never blocks the rest (spec 10)
    return out
