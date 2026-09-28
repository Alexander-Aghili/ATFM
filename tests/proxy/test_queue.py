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

async def test_waiting_interactive_is_promoted_when_slack_runs_out():
    now = [100.0]
    q = HoldQueue(window=1, clock=lambda: now[0], max_hold_s=600.0)
    filler = _entry("f", 0, 1.0, 99.0)
    q.submit(filler)                                   # takes the only slot
    waiting = Entry(session_id="i", tier=1, index=5.0, t_arrival=100.0, promote_at=104.0)
    q.submit(waiting)
    assert waiting.tier == 1
    now[0] = 105.0
    q.tick()
    assert waiting.tier == 2 and not waiting.released.is_set()
    q.complete(filler)
    assert waiting.released.is_set() and q.stats()["in_flight"] == 1
    q.complete(waiting); q.complete(waiting)           # idempotent per entry
    assert q.stats()["in_flight"] == 0


def test_batch_release_matches_repeated_selection():
    import random

    rng = random.Random(31)
    for overflow in (False, True):
        for available in (0, 1, 7, 100):
            queue = HoldQueue(window=available, clock=lambda: 10.0)
            queue.overflow = overflow
            entries = [Entry(str(i), rng.randrange(3), rng.randrange(4), rng.randrange(5),
                             not_before=rng.choice([0.0, 20.0]), promote_at=rng.choice([None, 5.0, 15.0]))
                       for i in range(60)]
            queue.pending = entries.copy()
            remaining, expected = entries.copy(), []
            for _ in range(available):
                eligible = [e for e in remaining if e.not_before <= 10.0]
                if not eligible:
                    break
                best = (min(eligible, key=lambda e: e.t_arrival) if overflow else
                        max(eligible, key=lambda e: (2 if e.tier == 1 and e.promote_at is not None and e.promote_at <= 10 else e.tier, e.index, -e.t_arrival)))
                remaining.remove(best)
                expected.append(best)
            queue.tick()
            assert list(queue.release_order) == [e.session_id for e in expected]
            assert queue.pending == remaining
            assert queue.in_flight == len(expected)
            assert all(e.released.is_set() and e.t_release == 10.0 for e in expected)
            assert all(not e.released.is_set() for e in remaining)
