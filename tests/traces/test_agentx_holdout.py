"""Evaluation roots and their sub-agent children never reach the board's training table."""
from atfm.schema.trace import TraceRow, TraceTable
from atfm.traces.agentx import exclude_traces


def row(sid, turn=0):
    return TraceRow(session_id=sid, cls="interactive", tenant="t", turn_index=turn, t_request=float(turn),
                    t_first_token=turn + .5, t_last_token=turn + 1.0, isl=10, osl=2, tool_name=None, source="test")


def test_excludes_roots_and_children_but_keeps_lookalike_ids():
    table = TraceTable.from_rows([row("root"), row("root", 1), row("root/agent-1"), row("rooted"), row("other")])
    kept = exclude_traces(table, ["root"])
    assert sorted(kept.df.session_id.unique()) == ["other", "rooted"]


def test_excluding_nothing_keeps_every_row():
    table = TraceTable.from_rows([row("a"), row("b")])
    assert len(exclude_traces(table, []).df) == 2
