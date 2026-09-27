"""Review fix pass for the v1-completion branch. One test per Critical/Important finding; each names the
production change that makes it pass."""
import json

import numpy as np
import pytest

from atfm.bus import InMemoryBus
from atfm.schema.trace import TraceRow, TraceTable


def _row(sid, k, t, isl=1000, cls="background", tool=None, tool_dur=None):
    kw = dict(session_id=sid, cls=cls, tenant="t", turn_index=k, t_request=t, t_first_token=t + 1, t_last_token=t + 2,
              isl=isl, osl=20, source="test")
    if tool:
        kw.update(tool_name=tool, backend_id="local", t_tool_start=t + 2, t_tool_end=t + 2 + tool_dur)
    return TraceRow(**kw)


# ---------------------------------------------------------------- C1: tool-less traces keep every turn
def test_trace_without_tool_columns_replays_every_call():
    from atfm.sim.core import Simulator
    from atfm.sim.engine import EngineConfig
    from atfm.sim.policies import NativePolicy
    from atfm.sim.programs import programs_from_table
    rows = [_row(f"s{i}", k, 100.0 * i + 30.0 * k) for i in range(4) for k in range(3)]
    progs = programs_from_table(TraceTable.from_rows(rows), None, 1000.0, np.random.default_rng(0))
    assert all(len(p.turns) == 3 for p in progs)
    assert [t.tool_name for t in progs[0].turns] == ["__gap__", "__gap__", None]      # gaps chain the calls
    assert progs[0].turns[0].tool_duration == pytest.approx(28.0) and progs[0].turns[0].think
    log = Simulator(progs, [EngineConfig(kv_blocks=3000, max_batch=4, prefill_tps=2e4, decode_tps=40.0)], NativePolicy(),
                    rng=np.random.default_rng(0)).run()
    assert len(log) == 12


# ---------------------------------------------------------------- C2: Redis bus never wedges on a bad entry
def test_redis_drain_skips_malformed_entries_and_advances_the_cursor(tmp_path):
    from atfm.bus.redis import RedisStreamsBus
    from atfm.schema.events import SessionStart
    from tests.bus.test_redis_bus import FakeRedis
    r = FakeRedis()
    bus = RedisStreamsBus(client=r, stream="atfm", cursor_path=tmp_path / "c")
    r.xadd("atfm", {"kind": "future.kind", "json": '{"kind": "future.kind", "t": 1.0}'})
    r.xadd("atfm", {"kind": "session.start", "json": "not json at all"})
    bus.publish(SessionStart(t=2.0, session_id="ok", tenant="t", cls="background"))
    got = bus.drain()
    assert [e.session_id for e in got] == ["ok"] and bus.malformed == 2
    assert bus.drain() == [] and (tmp_path / "c").read_text().strip() == bus.last_id
    assert not list(tmp_path.glob("c.*"))                                                 # atomic write leaves no temp file


def test_jsonl_read_events_skips_malformed_lines(tmp_path):
    from atfm.bus.jsonl import read_events
    from atfm.schema.events import SessionStart
    p = tmp_path / "e.jsonl"
    p.write_text(SessionStart(t=1.0, session_id="a", tenant="t", cls="background").model_dump_json() + "\n{bad\n"
                 + '{"kind": "future.kind", "t": 2.0}\n' + SessionStart(t=3.0, session_id="b", tenant="t", cls="background").model_dump_json() + "\n")
    assert [e.session_id for e in read_events(p)] == ["a", "b"]


# ---------------------------------------------------------------- I3: trace regime holds out sessions from training
def test_trace_regime_trains_on_held_out_sessions(tmp_path):
    from atfm_experiments.h2sim import H2SimConfig, _programs_for, split_table_for_training
    rows = [_row(f"s{i}", k, 100.0 * i + 30.0 * k, tool="bash", tool_dur=3.0) for i in range(10) for k in range(2)]
    table = TraceTable.from_rows(rows)
    path = tmp_path / "t.parquet"
    table.to_parquet(path)
    train, test = split_table_for_training(table, test_fraction=0.5, seed=0)
    assert set(train.df.session_id).isdisjoint(set(test.df.session_id)) and len(train.df) + len(test.df) == len(table.df)
    cfg = H2SimConfig(name="t", regime="trace", trace_path=str(path), duration_s=600.0, trace_rate_per_hour=120.0)
    tr = {p.session_id.split("#")[0] for p in _programs_for(cfg, 0, train=True)}
    te = {p.session_id.split("#")[0] for p in _programs_for(cfg, 0, train=False)}
    assert tr and te and tr.isdisjoint(te)


# ---------------------------------------------------------------- I4: GDP delay is measured from the session's own slot
def test_gdp_imposed_delay_is_relative_to_resumption_and_bounded_sessions_are_committed():
    from atfm.control import Deferrable, GdpPlanner
    from tests.control.test_gdp import _snap
    hz = [30.0 * k for k in range(1, 11)]
    empty = _snap(0.0, hz, [0] * 10, [0] * 10)
    planner = GdpPlanner(slot_s=30.0, horizon_s=300.0, eps=0.1, max_hold_s=600.0)
    a = Deferrable(session_id="a", tenant="t", eta_s=100.0, kv_blocks=10, prefill_tokens=10)
    out = planner.plan(0.0, empty, {"kv_blocks": 1000.0, "prefill_tokens": 1e9}, [a])
    assert out[0].release_not_before == 90.0 and planner.max_imposed_delay == {"t": 0.0}          # its own slot, no delay
    # slots 3..5 full: the session (slot 3) moves to slot 6 -> imposed 90 s; a tenant bound of 30 s releases at slot 4 and commits there
    full = _snap(0.0, hz, [0, 0, 0, 2000, 4000, 6000, 6000, 6000, 6000, 6000], [0] * 10)
    out = planner.plan(0.0, full, {"kv_blocks": 1000.0, "prefill_tokens": 1e9}, [a])
    assert out[0].release_not_before == 180.0 and planner.max_imposed_delay == {"t": 90.0}
    out = planner.plan(0.0, full, {"kv_blocks": 1000.0, "prefill_tokens": 1e9}, [a], tenant_max_delay={"t": 30.0})
    assert out[0].release_not_before == 120.0 and out[0].reason == "tenant_cap" and planner.last_assignment == {4: ["a"]}
    # the cap is measured from arrival too: eta 500 s, everything full, cap 600 -> release at 500 + 600
    b = Deferrable(session_id="b", tenant="t", eta_s=500.0, kv_blocks=10, prefill_tokens=10)
    planner2 = GdpPlanner(slot_s=30.0, horizon_s=300.0, eps=0.1, max_hold_s=600.0)
    out = planner2.plan(0.0, _snap(0.0, hz, list(np.cumsum([2000] * 10)), [0] * 10), {"kv_blocks": 1000.0, "prefill_tokens": 1e9}, [b])
    assert out[0].release_not_before == 1100.0 and out[0].reason == "capped"


# ---------------------------------------------------------------- I5: slot demand interpolates the cumulative forecast
def test_gdp_slot_demand_interpolates_between_horizons():
    from atfm.control import GdpPlanner
    from tests.control.test_gdp import _snap
    snap = _snap(0.0, [30.0, 120.0, 300.0], [100, 400, 1000], [0, 0, 0])
    planner = GdpPlanner(slot_s=30.0, horizon_s=300.0)
    d = [float(x.mean()) for x in planner.slot_demand(snap, "kv_blocks")]
    assert d == pytest.approx([100.0] * 10, abs=1e-6)                     # 100 per 30 s throughout, not [100,0,300,0,...]


# ---------------------------------------------------------------- I6: touch credit is capped at a burst
def test_touch_credit_never_exceeds_the_burst():
    from atfm.control import Residency, TouchController
    tc = TouchController(horizon_s=30.0, age_s=10.0, budget_per_s=1.0, tick_s=5.0)
    for t in range(120):                                                      # ten quiet minutes
        tc.plan(5.0 * t, {}, {}, {"w0": 100.0})
    res = {f"s{i}": Residency(worker_id="w0", blocks=10, last_used=-1000.0) for i in range(1000)}
    rs = {f"s{i}": (1.0, 2.0, 3.0) for i in range(1000)}
    assert len(tc.plan(600.0, rs, res, {"w0": 100.0})) == 5                  # one tick's worth, not 600
    tc2 = TouchController(budget_per_s=1.0, tick_s=5.0, burst=12)
    tc2.plan(0.0, {}, {}, {}); tc2.plan(5.0, {}, {}, {}); tc2.plan(10.0, {}, {}, {})
    assert len(tc2.plan(15.0, rs, res, {"w0": 100.0})) == 12


# ---------------------------------------------------------------- I7: wrapped executors never lose a finished result
def test_wrap_executor_returns_unknown_result_shapes_and_still_emits_tool_end():
    from atfm.sidecar.adapters import wrap_executor
    from atfm.sidecar.minisweagent import SidecarConfig

    class Obs:
        content, exit_code = "collected 1 items\nt::a PASSED [100%]\n", 0
    bus = InMemoryBus()
    cfg = SidecarConfig(session_id="w", cls="background", bus=bus)
    obs = Obs()
    assert wrap_executor(lambda c, **k: obs, cfg)("pytest") is obs
    assert wrap_executor(lambda c, **k: None, cfg, tuple_result=True)("pytest") is None
    kinds = [e.kind for e in bus.drain()]
    assert kinds.count("tool.start") == 2 and kinds.count("tool.end") == 2 and "tool.progress" not in kinds
    w = wrap_executor(lambda c, **k: obs, cfg, extract=lambda r: (r.content, r.exit_code))
    assert w("pytest") is obs
    ev = bus.drain()
    assert any(e.kind == "tool.progress" for e in ev) and ev[-1].exit_status == 0


# ---------------------------------------------------------------- I8: metrics from a worker's own page
def test_metrics_page_without_worker_labels_uses_the_default_worker_id():
    from atfm.board.metrics import worker_metrics_from_prometheus
    page = 'vllm:gpu_cache_usage_perc{model_name="m"} 0.25\nvllm:num_requests_waiting{model_name="m"} 4\n'
    evs = worker_metrics_from_prometheus(page, t=1.0, default_total_blocks=2000, default_worker_id="http://w0:8000")
    assert len(evs) == 1 and evs[0].worker_id == "http://w0:8000" and evs[0].kv_blocks_used == 500 and evs[0].queue_depth == 4
    dyn = 'dynamo_component_kv_total_blocks{dynamo_component="backend",dynamo_endpoint="generate"} 800\ndynamo_component_kv_active_blocks{dynamo_component="backend",dynamo_endpoint="generate"} 80\n'
    evs = worker_metrics_from_prometheus(dyn, t=1.0)
    assert len(evs) == 1 and evs[0].worker_id == "backend/generate" and evs[0].kv_blocks_used == 80


# ---------------------------------------------------------------- I9: proxy memory is bounded
async def test_proxy_last_body_and_release_order_are_bounded():
    import httpx
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from atfm.proxy.app import create_app
    from atfm.proxy.config import ProxyConfig
    from atfm.proxy.queue import HoldQueue
    up = FastAPI()

    @up.post("/v1/chat/completions")
    async def chat(req: Request):
        return JSONResponse({"choices": [{"message": {"content": "k"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    app = create_app(ProxyConfig(upstream_url="http://up", max_remembered_sessions=3),
                     upstream_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=up), base_url="http://up"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://p") as c:
        for i in range(6):
            await c.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
                         headers={"x-atfm-session": f"s{i}", "x-atfm-class": "background", "x-atfm-tenant": "t"})
    assert list(app.state.last_body) == ["s3", "s4", "s5"]
    q = HoldQueue(window=1, clock=lambda: 0.0, release_order_max=2)
    from atfm.proxy.queue import Entry
    for i in range(4):
        e = Entry(session_id=f"e{i}", tier=0, index=0.0, t_arrival=float(i))
        q.submit(e); q.complete(e)
    assert list(q.release_order) == ["e2", "e3"]


# ---------------------------------------------------------------- I10: as-recorded replay rebases time to zero
def test_trace_replay_without_overlay_rebases_timestamps():
    from atfm.sim.programs import programs_from_table
    rows = [_row("s", 0, 1.7e9, tool="bash", tool_dur=3.0), _row("s", 1, 1.7e9 + 10.0), _row("u", 0, 1.7e9 + 50.0, tool="bash", tool_dur=1.0), _row("u", 1, 1.7e9 + 60.0)]
    progs = programs_from_table(TraceTable.from_rows(rows), None, 1000.0, np.random.default_rng(0))
    assert [p.t_arrival for p in progs] == [0.0, 50.0]
