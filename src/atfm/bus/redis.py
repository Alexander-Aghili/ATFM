"""Redis Streams bus (deploy). Same `Bus` protocol as the in-memory and JSONL buses: `publish(event)` and
`drain() -> list[Event]`. The consumer cursor (last stream id) can be persisted so a restarted board or
proxy resumes where it left off instead of replaying or skipping."""
from __future__ import annotations

import json
from pathlib import Path

from atfm.schema.events import Event, parse_event


class RedisStreamsBus:
    def __init__(self, client=None, url: str | None = None, stream: str = "atfm.events", cursor_path: str | Path | None = None,
                 maxlen: int | None = 1_000_000, batch: int = 10_000):
        if client is None:
            try:
                import redis  # noqa: F401
            except ImportError as e:
                raise RuntimeError("the redis package is required for RedisStreamsBus with a URL (uv add redis)") from e
            import redis as _redis
            client = _redis.Redis.from_url(url or "redis://localhost:6379/0")
        self.client, self.stream, self.maxlen, self.batch = client, stream, maxlen, batch
        self.cursor_path = Path(cursor_path) if cursor_path else None
        self.last_id = "0"
        if self.cursor_path and self.cursor_path.exists():
            self.last_id = self.cursor_path.read_text().strip() or "0"
        self.dropped = 0

    def publish(self, e: Event) -> None:
        try:
            self.client.xadd(self.stream, {"kind": e.kind, "json": e.model_dump_json()}, maxlen=self.maxlen, approximate=True)
        except Exception:
            self.dropped += 1                     # fail-open (spec 10): a bus failure never blocks the caller

    def drain(self) -> list[Event]:
        out: list[Event] = []
        res = self.client.xread({self.stream: self.last_id}, count=self.batch)
        for _name, entries in res or []:
            for eid, fields in entries:
                raw = fields.get(b"json", fields.get("json"))
                if isinstance(raw, bytes):
                    raw = raw.decode()
                out.append(parse_event(json.loads(raw)))
                self.last_id = eid.decode() if isinstance(eid, bytes) else eid
        if self.cursor_path is not None and out:
            self.cursor_path.write_text(self.last_id)
        return out
