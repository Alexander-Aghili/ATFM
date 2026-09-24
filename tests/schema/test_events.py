from atfm.schema.events import parse_event, event_to_dict, ToolProgress, LlmRequest

def test_roundtrip_by_kind():
    d = {"kind": "tool.progress", "t": 1.5, "session_id": "s", "call_id": "c", "completed": 3, "total": 10, "phase": "run"}
    e = parse_event(d)
    assert isinstance(e, ToolProgress) and e.completed == 3
    assert event_to_dict(e) == d

def test_class_alias_and_defaults():
    e = parse_event({"kind": "session.start", "t": 0.0, "session_id": "s", "tenant": "t", "class": "background"})
    assert e.cls == "background" and e.parent_session_id is None and e.deadline is None
    r = parse_event({"kind": "llm.request", "t": 2.0, "session_id": "s", "turn_index": 0, "request_id": "r", "isl": 100})
    assert isinstance(r, LlmRequest) and r.hints == {} and r.held_s == 0.0

def test_unknown_kind_rejected():
    import pytest
    with pytest.raises(ValueError):
        parse_event({"kind": "nope", "t": 0.0})
