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


def test_jsonl_incremental_independent_readers_and_restart(tmp_path):
    p = tmp_path / "events.jsonl"
    writer, reader = JsonlBus(p), JsonlBus(p)
    assert reader.drain() == []
    writer.publish(_ev()[0])
    assert reader.drain() == [_ev()[0]]
    assert reader.drain() == []
    writer.publish(_ev()[1])
    assert reader.drain() == [_ev()[1]]
    assert writer.drain() == _ev()
    assert JsonlBus(p).drain() == _ev()
    assert read_events(p) == _ev()


def test_jsonl_partial_utf8_and_malformed_complete_lines(tmp_path):
    import json
    from atfm.schema.events import event_to_dict

    p = tmp_path / "events.jsonl"
    bus = JsonlBus(p)
    event = _ev()[0].model_copy(update={"tool_name": "café"})
    line = (json.dumps(event_to_dict(event), ensure_ascii=False) + "\n").encode()
    split = line.index(b"\xc3") + 1
    p.write_bytes(b'garbage\n{}\n\xff\n\n' + line[:split])
    assert bus.drain() == []
    assert bus.malformed == 3
    assert bus.drain() == []
    assert bus.malformed == 3
    with p.open('ab') as stream:
        stream.write(line[split:-1])
    assert bus.drain() == []
    with p.open('ab') as stream:
        stream.write(b'\n')
    assert bus.drain() == [event]
    assert bus.drain() == []


def test_jsonl_replacement_missing_path_and_observed_truncation(tmp_path):
    p = tmp_path / "events.jsonl"
    bus = JsonlBus(p)
    bus.publish(_ev()[0])
    assert bus.drain() == [_ev()[0]]
    p.rename(tmp_path / "old.jsonl")
    assert bus.drain() == []
    bus.publish(_ev()[1])
    assert bus.drain() == [_ev()[1]]
    p.write_bytes(b'')
    assert bus.drain() == []
    bus.publish(_ev()[0])
    assert bus.drain() == [_ev()[0]]


def test_jsonl_io_failure_does_not_advance_cursor(tmp_path, monkeypatch):
    from pathlib import Path
    from unittest.mock import MagicMock
    import pytest

    p = tmp_path / "events.jsonl"
    bus = JsonlBus(p)
    for event in _ev():
        bus.publish(event)
    with p.open('rb') as stream:
        failing = MagicMock(wraps=stream)
        failing.__enter__.return_value = failing
        failing.readline.side_effect = [stream.readline(), OSError('disk read')]
        with monkeypatch.context() as patch:
            patch.setattr(Path, 'open', lambda *a, **k: failing)
            with pytest.raises(OSError):
                bus.drain()
    assert bus.drain() == _ev()
    assert bus.drain() == []
