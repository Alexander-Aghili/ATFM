"""Redis Streams bus (deploy): same Bus protocol as the in-memory and JSONL buses, resumable consumer."""
import pytest

from atfm.bus.redis import RedisStreamsBus
from atfm.schema.events import LlmRequest, SessionStart, ToolStart


class FakeRedis:
    """The subset of redis-py used by the bus: xadd, xread (blocking ignored), xrange."""

    def __init__(self):
        self.streams: dict[str, list[tuple[str, dict]]] = {}
        self._seq = 0

    def xadd(self, name, fields, maxlen=None, approximate=True):
        self._seq += 1
        eid = f"{1000 + self._seq}-0"
        self.streams.setdefault(name, []).append((eid, {k: (v.encode() if isinstance(v, str) else v) for k, v in fields.items()}))
        return eid

    def xread(self, streams, count=None, block=None):
        out = []
        for name, last in streams.items():
            entries = [(e, f) for e, f in self.streams.get(name, []) if last == "0" or _gt(e, last)]
            if count:
                entries = entries[:count]
            if entries:
                out.append((name.encode(), entries))
        return out


def _gt(a: str, b: str) -> bool:
    return tuple(int(x) for x in a.split("-")) > tuple(int(x) for x in b.split("-"))


def test_publish_and_drain_round_trip_preserves_models():
    r = FakeRedis()
    bus = RedisStreamsBus(client=r, stream="atfm")
    evs = [SessionStart(t=1.0, session_id="s", tenant="t", cls="background"),
           LlmRequest(t=2.0, session_id="s", turn_index=0, request_id="r", isl=10),
           ToolStart(t=3.0, session_id="s", turn_index=0, call_id="c", tool_name="pytest", args_hash="sig")]
    for e in evs:
        bus.publish(e)
    got = bus.drain()
    assert got == evs and bus.drain() == []
    bus.publish(SessionStart(t=4.0, session_id="s2", tenant="t", cls="interactive"))
    assert [e.session_id for e in bus.drain()] == ["s2"]


def test_consumer_resumes_from_a_persisted_cursor(tmp_path):
    r = FakeRedis()
    cur = tmp_path / "cursor"
    a = RedisStreamsBus(client=r, stream="atfm", cursor_path=cur)
    a.publish(SessionStart(t=1.0, session_id="s1", tenant="t", cls="background"))
    a.publish(SessionStart(t=2.0, session_id="s2", tenant="t", cls="background"))
    assert len(a.drain()) == 2
    b = RedisStreamsBus(client=r, stream="atfm", cursor_path=cur)   # restarted consumer
    assert b.drain() == []
    b.publish(SessionStart(t=3.0, session_id="s3", tenant="t", cls="background"))
    assert [e.session_id for e in b.drain()] == ["s3"]


def test_missing_redis_package_is_a_clear_error_only_with_a_real_url(monkeypatch):
    import builtins
    real = builtins.__import__
    def fake(name, *a, **k):
        if name == "redis":
            raise ImportError("no redis")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(RuntimeError, match="redis"):
        RedisStreamsBus(url="redis://localhost:6379/0")
    assert RedisStreamsBus(client=FakeRedis()).drain() == []
