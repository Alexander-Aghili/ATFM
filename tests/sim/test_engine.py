from atfm.sim.engine import EngineConfig, Worker, Request, Router

def _req(rid, sid, isl_total, isl_new, osl, tier=0, index=1.0, t=0.0, cls="background"):
    return Request(request_id=rid, session_id=sid, cls=cls, isl_total=isl_total, isl_new=isl_new, osl=osl, tier=tier, index=index, t_queued=t)

def test_prefix_hit_and_recompute_after_eviction():
    w = Worker("w0", EngineConfig(kv_blocks=100, max_batch=4, prefill_tps=1000.0, decode_tps=10.0))
    w.submit(_req("r1", "a", 160, 160, 16), 0.0)
    (req, ts, tf, te, hit, recomputed, ev), = w.schedule(0.0)
    assert hit == 0 and recomputed == 160 and abs(tf - 0.16) < 1e-9 and abs(te - 0.16 - 1.6) < 1e-9
    w.complete("r1", te)
    assert w.resident_blocks("a") == 11
    w.submit(_req("r2", "a", 240, 64, 16), 5.0)
    (_, _, _, _, hit, recomputed, _), = w.schedule(5.0)
    assert hit == 176 and recomputed == 64
    w.complete("r2", 6.0)
    w.evict_session("a")
    w.submit(_req("r3", "a", 320, 64, 16), 7.0)
    (_, _, _, _, hit, recomputed, _), = w.schedule(7.0)
    assert hit == 0 and recomputed == 320

def test_waits_when_no_evictable_blocks():
    w = Worker("w0", EngineConfig(kv_blocks=20, max_batch=4, prefill_tps=1000.0, decode_tps=10.0))
    w.submit(_req("r1", "a", 160, 160, 16), 0.0)
    w.schedule(0.0)
    w.submit(_req("r2", "b", 160, 160, 16), 0.0)
    assert w.schedule(0.0) == [] and len(w.queue) == 1
    w.complete("r1", 2.0)
    out = w.schedule(2.0)
    assert len(out) == 1 and out[0][6] == 1 and w.resident_blocks("a") == 0

def test_priority_ordering_and_batch_limit():
    w = Worker("w0", EngineConfig(kv_blocks=1000, max_batch=1, prefill_tps=1000.0, decode_tps=10.0, priority=True))
    w.submit(_req("bg", "s1", 16, 16, 1, tier=0, index=1.0), 0.0)
    w.submit(_req("it", "s2", 16, 16, 1, tier=1, index=0.5, cls="interactive"), 0.0)
    out = w.schedule(0.0)
    assert [o[0].request_id for o in out] == ["it"] and len(w.queue) == 1

def test_router_affinity():
    ws = [Worker(f"w{i}", EngineConfig(kv_blocks=100, max_batch=4, prefill_tps=1000.0, decode_tps=10.0)) for i in range(2)]
    r = Router(ws, "affinity")
    w = r.pick(_req("r1", "a", 16, 16, 1)); w.submit(_req("r1", "a", 16, 16, 1), 0.0); w.schedule(0.0); w.complete("r1", 1.0)
    assert r.pick(_req("r2", "a", 32, 16, 1)) is w
    other = r.pick(_req("r3", "b", 16, 16, 1))
    assert other is not w
