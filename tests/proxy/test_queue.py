import asyncio, pytest
from atfm.proxy.queue import HoldQueue, Entry

def _entry(sid, tier, index, t):
    return Entry(session_id=sid, tier=tier, index=index, t_arrival=t)

async def test_window_and_priority_order():
    now = [100.0]
    q = HoldQueue(window=1, clock=lambda: now[0], max_hold_s=600.0)
    a, b, c, d = _entry("a", 0, 1.0, 1.0), _entry("b", 0, 2.0, 2.0), _entry("c", 0, 3.0, 3.0), _entry("d", 2, 0.5, 4.0)
    for e in (a, b, c, d):
        q.submit(e)
    await asyncio.wait_for(a.released.wait(), 1.0)
    assert not b.released.is_set() and not d.released.is_set()
    q.complete()
    await asyncio.wait_for(d.released.wait(), 1.0)
    assert not c.released.is_set()
    q.complete()
    await asyncio.wait_for(c.released.wait(), 1.0)
    assert not b.released.is_set()

async def test_directive_hold_and_cap():
    now = [100.0]
    q = HoldQueue(window=4, clock=lambda: now[0], max_hold_s=10.0)
    q.set_directive("h", release_not_before=100000.0, reason="gdp")
    e = _entry("h", 0, 1.0, 100.0)
    q.submit(e)
    assert not e.released.is_set() and q.stats()["held"] == 1
    assert e.not_before == 110.0
    now[0] = 111.0
    q.tick()
    await asyncio.wait_for(e.released.wait(), 1.0)
