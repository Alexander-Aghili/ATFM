import asyncio
import random
import time

import pytest

from atfm.proxy.queue import Entry, HoldQueue
from tests.proxy.reference.queue_scan import Entry as ScanEntry, HoldQueue as ScanQueue


@pytest.mark.parametrize('seed', range(8))
def test_indexed_queue_matches_scan_reference_over_event_traces(seed):
    rng = random.Random(seed)
    now = [0.0]
    queues = [cls(3, clock=lambda: now[0], max_hold_s=8, max_size=15, release_order_max=10000)
              for cls in (HoldQueue, ScanQueue)]
    entries = [[], []]
    for step in range(1000):
        action = rng.choice(['submit'] * 5 + ['complete'] * 3 + ['cancel', 'advance', 'window'])
        _apply_action(action, rng, now, queues, entries, step)
        _assert_equivalent(queues, entries, seed, step, action)
    for queue in queues:
        queue.window = 10000
        now[0] += 100
        queue.tick()
    assert list(queues[0].release_order) == list(queues[1].release_order)
    assert queues[0].queued == 0


def _assert_equivalent(queues, entries, seed, step, action):
    assert queues[0].stats() == queues[1].stats(), (seed, step, action)
    _assert_peer_ranks(queues)
    assert list(queues[0].release_order) == list(queues[1].release_order)
    for left, right in zip(entries[0], entries[1]):
        assert (left.tier, left.not_before, left.t_release, left.done) == (right.tier, right.not_before, right.t_release, right.done)
    counterparts = {id(right): id(left) for left, right in zip(*entries)}
    assert [id(e) for e in queues[0].pending] == [counterparts[id(e)] for e in queues[1].pending]


def _apply_action(action, rng, now, queues, entries, step):
    if action == 'submit':
        _submit_pair(rng, now, queues, entries, step)
    elif action in ('complete', 'cancel') and entries[0]:
        index = rng.randrange(len(entries[0]))
        for j, queue in enumerate(queues):
            getattr(queue, action)(entries[j][index])
    elif action == 'advance':
        now[0] += rng.choice([0, .5, 2, 10])
        for queue in queues:
            queue.tick()
    elif action == 'window':
        window = rng.randrange(6)
        for queue in queues:
            queue.window = window
            queue.tick()


def _submit_pair(rng, now, queues, entries, step):
    sid = f's{step % 17}'
    tier, index = rng.randrange(3), rng.randrange(4)
    promotion = now[0] + rng.choice([0, 1, 5]) if tier == 1 else None
    hold = now[0] + rng.randrange(12)
    expires = now[0] + rng.choice([-1, 0, 10])
    for j, (queue, cls) in enumerate(zip(queues, (Entry, ScanEntry))):
        queue.set_directive(sid, hold, 'test', expires)
        e = cls(sid, tier, index, now[0], promote_at=promotion)
        entries[j].append(e)
        queue.submit(e)


def test_equal_keys_keep_insertion_order_and_compact_cancelled_records():
    queue = HoldQueue(0, clock=lambda: 100)
    entries = [Entry(str(i), 1, 1, 100, not_before=200, promote_at=150) for i in range(500)]
    for e in entries:
        queue.submit(e)
    for e in entries[:-2]:
        queue.cancel(e)
    assert sum(map(len, (queue._ready, queue._arrival, queue._delayed, queue._promotions))) <= 8 * queue.queued + 64
    queue.clock = lambda: 201
    queue.window = 2
    queue.tick()
    assert list(queue.release_order) == ['498', '499']
    assert all(e.tier == 2 for e in entries[-2:])


async def test_hold_and_promotion_timers_fire_without_external_traffic():
    queue = HoldQueue(0, clock=time.monotonic)
    now = time.monotonic()
    e = Entry('s', 1, 1, now, not_before=now + .06, promote_at=now + .01)
    queue.submit(e)
    await asyncio.sleep(.03)
    assert e.tier == 2 and not e.released.is_set()
    queue.window = 1
    await asyncio.wait_for(e.released.wait(), 1)
    assert queue._timer is None


async def test_cancellation_clears_last_timer():
    queue = HoldQueue(1, clock=time.monotonic)
    e = Entry('s', 1, 1, time.monotonic(), not_before=time.monotonic() + 100)
    queue.submit(e)
    assert queue._timer is not None
    queue.cancel(e)
    assert queue._timer is None and queue.queued == 0


def _assert_peer_ranks(queues):
    from tests.proxy.test_peer_ranks import scan_bucket
    for tier in range(3):
        indices = [e.index for e in queues[1].pending if e.tier == tier]
        for index in (-1., 0., 1.5, 3., 4.):
            assert queues[0].priority_bucket(tier, index) == scan_bucket(indices, index)
