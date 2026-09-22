import gzip, json
from atfm.traces.tracelab import rounds_to_rows, load_tracelab

def _ev(t, typ, src, **kw):
    return {"event_type": typ, "timestamp": t, "source": src, **kw}

def _round(idx, events, tools, isl=1000, osl=50):
    return {"provider": "claude", "session_id": "claude:abc", "round_index": idx, "round_id": f"r{idx}",
            "model": "m", "input_tokens_total": isl, "prefix_tokens": isl // 2, "newly_append_tokens": isl // 2,
            "output_tokens": osl, "timing_events": events, "tools": tools, "user": "user_1",
            "first_input_event_type": events[0]["event_type"] if events else None}

def _tool(name, emitted, result, ms, err=False, cid="c1"):
    return {"tool_index": 0, "tool_name": name, "tool_call_id": cid, "emitted_at": emitted,
            "input_chars": 10, "result_chars": 20, "tool_wall_latency_ms": ms,
            "tool_internal_latency_ms": None, "is_error": err, "result_at": result}

T0 = "2026-04-18T08:23:07.000Z"; T1 = "2026-04-18T08:23:09.000Z"; T2 = "2026-04-18T08:23:10.000Z"
T3 = "2026-04-18T08:23:12.000Z"; T4 = "2026-04-18T08:23:15.000Z"; T5 = "2026-04-18T08:23:20.000Z"; T6 = "2026-04-18T08:24:00.000Z"

def test_three_round_session():
    rounds = [
        _round(0, [_ev(T0, "user_message", "user.message"), _ev(T1, "text", "assistant.content.text"),
                   _ev(T2, "tool_call", "assistant.content.tool_use")],
               [_tool("Bash", T2, T3, 2000)]),
        _round(1, [_ev(T3, "tool_result", "user.content.tool_result"), _ev(T4, "text", "assistant.content.text")], []),
        _round(2, [_ev(T6, "user_message", "user.message"), _ev(T6, "text", "assistant.content.text")], []),
    ]
    rows = rounds_to_rows(rounds)
    assert [r.turn_index for r in rows] == [0, 1, 2]
    r0, r1, r2 = rows
    assert r0.tool_name == "Bash" and r0.t_tool_end - r0.t_tool_start == 2.0
    assert r0.t_first_token < r0.t_last_token and r0.prefix_hit_tokens == 500
    assert r1.tool_name == "__think__" and r1.t_tool_start == r1.t_last_token
    assert abs(r1.t_tool_end - r2.t_request) < 1e-6
    assert r2.tool_name is None and r2.t_tool_start is None
    assert all(r.cls == "interactive" and r.tenant == "user_1" for r in rows)

def test_parallel_tools_critical_path():
    rounds = [_round(0, [_ev(T0, "tool_result", "user.content.tool_result"), _ev(T1, "tool_call", "assistant.content.tool_use")],
                     [_tool("Read", T1, T2, 1000, cid="a"), _tool("Bash", T1, T4, 6000, cid="b"), _tool("Read", T2, T3, 2000, cid="c")])]
    r = rounds_to_rows(rounds)[0]
    assert r.tool_name == "Bash" and r.t_tool_end - r.t_tool_start == 6.0 and r.tool_exit_status == 0

def test_error_and_spawn_flags():
    rounds = [_round(0, [_ev(T0, "user_message", "user.message"), _ev(T1, "tool_call", "assistant.content.tool_use")],
                     [_tool("Agent", T1, T5, 11000, err=True)])]
    r = rounds_to_rows(rounds)[0]
    assert r.tool_exit_status == 1 and r.spawned_children == 1

def test_load_gz(tmp_path):
    p = tmp_path / "t.jsonl.gz"
    with gzip.open(p, "wt") as f:
        for rr in [_round(1, [_ev(T3, "tool_result", "user.content.tool_result")], []),
                   _round(0, [_ev(T0, "user_message", "user.message"), _ev(T2, "tool_call", "assistant.content.tool_use")], [_tool("Bash", T2, T3, 2000)])]:
            f.write(json.dumps(rr) + "\n")
    t = load_tracelab(p)
    assert len(t) == 2 and t.df["turn_index"].tolist() == [0, 1] and t.df["source"].iloc[0] == "tracelab"
