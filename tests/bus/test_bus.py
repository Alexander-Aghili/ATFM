from atfm.bus import InMemoryBus, JsonlBus, read_events
from atfm.schema.events import ToolStart, ToolEnd

def _ev():
    return [ToolStart(t=1.0, session_id="s", turn_index=0, call_id="c", tool_name="pytest", backend_id="local"),
            ToolEnd(t=5.0, session_id="s", call_id="c", exit_status=0, output_chars=12)]

def test_memory_bus_drain_clears():
    b = InMemoryBus()
    for e in _ev():
        b.publish(e)
    assert [e.kind for e in b.drain()] == ["tool.start", "tool.end"] and b.drain() == []

def test_jsonl_bus_roundtrip(tmp_path):
    p = tmp_path / "ev.jsonl"
    b = JsonlBus(p)
    for e in _ev():
        b.publish(e)
    back = read_events(p)
    assert [e.kind for e in back] == ["tool.start", "tool.end"] and back[1].output_chars == 12
