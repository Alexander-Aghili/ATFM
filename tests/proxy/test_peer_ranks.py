"""Histogram ranks match the original list/NumPy definition, including ties."""
import numpy as np
import pytest

from atfm.proxy.index import priority_bucket
from atfm.proxy.queue import Entry, HoldQueue


def scan_bucket(values, index):
    return min(3, int(float(np.mean(np.asarray(values) < index)) * 4)) if values else 3


@pytest.mark.parametrize('values', [[], [1.]*100, [0., 1., 1., 2.], [float('nan'), 0., float('inf'), -float('inf')]])
@pytest.mark.parametrize('index', [-float('inf'), -1., 0., 1., 1.5, 2., float('inf'), float('nan')])
def test_peer_rank_parity_for_ties_boundaries_and_nonfinite_values(values, index):
    queue = HoldQueue(0, clock=lambda: 0.)
    queue.pending = [Entry(str(i), 0, value, 0.) for i, value in enumerate(values)]
    expected = scan_bucket(values, index)
    assert queue.priority_bucket(0, index) == priority_bucket(index, values) == expected
    assert queue.priority_bucket(1, index) == 3
    for entry in list(queue.pending):
        queue.cancel(entry)
    assert queue._ranks.counts == {} and queue._ranks.totals == {}


def test_duplicate_histogram_stays_small_and_rebuild_replaces_old_counts():
    queue = HoldQueue(0, clock=lambda: 0.)
    queue.pending = [Entry(str(i), i % 3, float(i % 5), 0.) for i in range(10000)]
    assert sum(len(c) for c in queue._ranks.counts.values()) == 15
    assert sum(queue._ranks.totals.values()) == 10000
    queue.pending = [Entry('replacement', 2, 99., 0.)]
    assert queue.priority_bucket(0, 100.) == 3
    assert queue.priority_bucket(2, 99.) == 0
    queue.window = 1
    queue.tick()
    assert queue.priority_bucket(2, 0.) == 3
    assert not queue._ranks.counts


def test_nan_removal_and_signed_zero_share_exact_rank_semantics():
    queue = HoldQueue(0, clock=lambda: 0.)
    entries = [Entry(str(i), 0, value, 0.) for i, value in enumerate([float('nan'), float('nan'), -0., 0., 1.])]
    queue.pending = entries
    queue.cancel(entries[0])
    assert queue.priority_bucket(0, 1.) == scan_bucket([float('nan'), -0., 0., 1.], 1.)
    queue.cancel(entries[2])
    assert queue.priority_bucket(0, 1.) == scan_bucket([float('nan'), 0., 1.], 1.)
