"""Proxy-only sessions: a finished call starts the corpus's `__gap__` phase so the gap model applies."""
from atfm.board.live import SessionRegistry
from atfm.schema.events import LlmDone, LlmRequest, SessionStart
from atfm.traces.agentx import GAP


def replay(registry, events):
    for e in events:
        registry.apply(e)
    return registry.get("a")


EVENTS = [SessionStart(t=0.0, session_id="a", tenant="t", cls="background"),
          LlmRequest(t=1.0, session_id="a", turn_index=0, request_id="r1", isl=1000),
          LlmDone(t=3.0, session_id="a", request_id="r1", osl=20)]


def test_llm_done_starts_a_gap_tool_phase_when_enabled():
    s = replay(SessionRegistry(gap_after_done=True), EVENTS)
    assert (s.phase, s.tool_name, s.t_tool_start) == ("tool_running", GAP, 3.0)
    assert s.elapsed(10.0) == 7.0


def test_next_call_records_the_gap_in_tool_history():
    s = replay(SessionRegistry(gap_after_done=True),
               EVENTS + [LlmRequest(t=12.5, session_id="a", turn_index=1, request_id="r2", isl=1100)])
    assert s.phase == "llm_running" and s.tool_history == [(GAP, 9.5)]


def test_default_registry_keeps_the_pending_phase():
    s = replay(SessionRegistry(), EVENTS)
    assert s.phase == "llm_pending" and s.tool_name is None
