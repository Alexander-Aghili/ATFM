import gzip, json
import pandas as pd
from atfm.traces.agentx import trace_to_rows, load_agentx

def _req(t, inp, out, api, ttft=None, typ="s", hash_ids=None):
    r = {"t": t, "model": "m", "in": inp, "out": out, "hash_ids": hash_ids or list(range(inp // 64)), "api_time": api, "type": typ}
    if ttft is not None:
        r["ttft"] = ttft
    return r

def _trace():
    return {"id": "abc", "block_size": 64, "models": ["m"], "requests": [
        _req(0.0, 640, 20, 2.0, ttft=1.0),                       # turn 0, then a 10 s gap
        _req(12.0, 1280, 30, 3.0, ttft=1.5),                     # turn 1, then a subagent group starts in its gap
        {"t": 16.0, "type": "subagent", "agent_id": "subagent_001", "subagent_type": "Subagent", "duration_ms": 8000,
         "total_tokens": 100, "tool_use_count": 2, "status": "completed", "models": ["m"], "requests": [
             _req(16.0, 320, 10, 1.0, typ="n"), _req(20.0, 384, 10, 1.0, typ="n")]},
        _req(30.0, 1920, 40, 2.0, ttft=1.0),                     # turn 2, last
    ]}

def test_trace_to_rows_main_agent_gaps_and_children():
    rows = trace_to_rows(_trace())
    main = sorted([r for r in rows if r.session_id == "abc"], key=lambda r: r.turn_index)
    assert [r.turn_index for r in main] == [0, 1, 2]
    r0, r1, r2 = main
    assert r0.t_request == 0.0 and r0.t_first_token == 1.0 and r0.t_last_token == 2.0 and r0.isl == 640 and r0.osl == 20
    assert r0.tool_name == "__gap__" and r0.t_tool_start == 2.0 and r0.t_tool_end == 12.0 and r0.backend_id == "unknown"
    assert r1.tool_name == "Agent" and r1.spawned_children == 1 and r1.t_tool_start == 15.0 and r1.t_tool_end == 30.0
    assert r2.tool_name is None and r2.cls == "interactive" and r2.source == "agentx"
    assert r1.prefix_hit_tokens == 640                             # first 10 blocks shared with turn 0
    child = sorted([r for r in rows if r.session_id == "abc/subagent_001"], key=lambda r: r.turn_index)
    assert len(child) == 2 and child[0].parent_session_id == "abc" and child[0].t_request == 16.0
    assert child[0].tool_name == "__gap__" and child[0].t_tool_end == 20.0 and child[1].tool_name is None

def test_negative_gap_clamped_and_load(tmp_path):
    tr = _trace()
    tr["requests"][1]["t"] = 1.5                                   # overlaps the previous response end (2.0)
    rows = trace_to_rows(tr)
    r0 = [r for r in rows if r.session_id == "abc" and r.turn_index == 0][0]
    assert r0.t_tool_end == r0.t_tool_start == 2.0
    p = tmp_path / "traces.jsonl"
    with open(p, "w") as f:
        f.write(json.dumps(_trace()) + "\n"); f.write(json.dumps({**_trace(), "id": "def"}) + "\n")
    t = load_agentx(p, limit=1)
    assert t.df.session_id.str.startswith("abc").all() and len(t) == 5
