"""Numerical, ordering, and random-stream invariants for CPU optimizations."""
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from atfm.eval.serving import MeanMetric, _paired_draws
from atfm.sim.engine import EngineConfig, Request, Worker


def reference_draws(per_session, metric, n_boot, rng):
    ids = sorted(set.intersection(*[set(frame.session_id) for frame in per_session.values()]))
    aligned = {arm: frame.set_index("session_id").loc[ids] for arm, frame in per_session.items()}
    result = {arm: [] for arm in per_session}
    for _ in range(n_boot):
        sample = rng.choice(ids, size=len(ids), replace=True)
        for arm in per_session:
            result[arm].append(metric(aligned[arm].loc[sample].reset_index()))
    return result


@pytest.mark.parametrize("size", [0, 1, 17])
@pytest.mark.parametrize("optimized", [False, True])
def test_bootstrap_preserves_pairing_missing_values_and_rng(size, optimized):
    frame = pd.DataFrame({"session_id": [f"s{i}" for i in range(size)], "value": np.arange(size, dtype=float)})
    if size > 1:
        frame.loc[0, "value"] = np.nan
    other = frame.iloc[::-1].assign(value=lambda d: d.value + 2)
    other = pd.concat([other, pd.DataFrame({"session_id": ["unpaired"], "value": [999.0]})], ignore_index=True)
    per = {"a": frame, "b": other}
    legacy = lambda d: 1.0 - d.value.mean()
    metric = MeanMetric(lambda d: d.value, offset=1.0, scale=-1.0) if optimized else legacy
    rng, old_rng = np.random.default_rng(42), np.random.default_rng(42)
    actual, _ = _paired_draws(per, metric, 20, rng)
    expected = reference_draws(per, legacy, 20, old_rng)
    for arm in per:
        np.testing.assert_allclose(actual[arm], expected[arm], rtol=1e-14, atol=1e-14, equal_nan=True)
    assert rng.random() == old_rng.random()


def test_generic_bootstrap_still_supports_nonlinear_metrics_and_duplicate_ids():
    frame = pd.DataFrame({"session_id": ["a", "b", "a"], "value": [1.0, 5.0, 20.0]})
    per = {"a": frame, "b": frame.assign(value=frame.value * 2)}
    metric = lambda d: d.value.median()
    actual, _ = _paired_draws(per, metric, 20, np.random.default_rng(3))
    assert actual == reference_draws(per, metric, 20, np.random.default_rng(3))


def reference_room(worker, needed, keep):
    victims = []
    protected = worker._running_sessions() | set(worker.pins)
    order = None
    if worker.victim_policy is not None:
        order = iter(worker.victim_policy([sid for sid in worker.resident if sid not in protected and sid != keep]))
    while worker.free_blocks() < needed:
        if order is None:
            victim = next((sid for sid in worker.resident if sid not in protected and sid != keep), None)
        else:
            victim = next((sid for sid in order if sid in worker.resident), None)
        if victim is None:
            return None
        del worker.resident[victim]
        victims.append(victim)
    return victims


@pytest.mark.parametrize("custom_order", [False, True])
def test_eviction_preserves_order_protection_and_partial_failure(custom_order):
    rng = np.random.default_rng(0)
    for _ in range(40):
        blocks = rng.integers(1, 10, 20)
        workers = [Worker("w", EngineConfig(int(blocks.sum()), 4, 1000, 40)) for _ in range(2)]
        for worker in workers:
            worker.resident = OrderedDict((f"s{i}", int(n)) for i, n in enumerate(blocks))
            worker.pins = {"s0": 10.0, "s2": 10.0}
            req = Request("r", "s1", "background", 10, 10, 1)
            worker.running["r"] = (req, 0, 10)
            if custom_order:
                worker.victim_policy = lambda candidates: list(reversed(candidates))
        need = int(rng.integers(0, int(blocks.sum()) + 20))
        assert workers[0]._make_room(need, "s3") == reference_room(workers[1], need, "s3")
        assert workers[0].resident == workers[1].resident


def test_conditional_duration_cache_preserves_draws_and_refitting():
    from atfm.board.predictors.duration import DurationModel
    from atfm.schema.trace import TraceRow, TraceTable

    model = DurationModel()
    for shift in (0.0, 10.0):
        table = TraceTable.from_rows([
            TraceRow(session_id=f"s{i}", cls="background", tenant="t", turn_index=0,
                     t_request=0, t_last_token=shift + duration, isl=16, osl=1, source="test")
            for i, duration in enumerate([2.0, 8.0, 3.0, 1.0, 9.0, 6.0, 7.0])
        ])
        model.fit(table)
        for cls in ("background", "unknown"):
            for elapsed in (0.0, 5.0, 100.0):
                rng, legacy_rng = np.random.default_rng(1), np.random.default_rng(1)
                expected = model._conditional(np.sort(model._llm["background"]), elapsed, 32,
                                              legacy_rng, model.min_conditional)
                actual = model.llm_duration_conditional(cls, elapsed, 32, rng)
                np.testing.assert_array_equal(actual, expected)
                assert rng.random() == legacy_rng.random()


def test_full_worker_defers_queue_ordering_until_a_slot_opens():
    worker = Worker("w", EngineConfig(1000, 1, 1000, 40, priority=True))
    worker.submit(Request("running", "running", "background", 16, 16, 1), 0.0)
    worker.schedule(0.0)
    for i in range(20):
        worker.submit(Request(str(i), str(i), "background", 16, 16, 1, tier=i % 2), float(i))
        worker.last_evictions = [("stale", "victim")]
        assert worker.schedule(float(i)) == []
        assert worker.last_evictions == []
    admitted = []
    worker.complete("running", 20.0)
    while worker.queue:
        (request, *_) , = worker.schedule(21.0)
        admitted.append(request.request_id)
        worker.complete(request.request_id, 22.0)
    assert admitted == [str(i) for i in range(1, 20, 2)] + [str(i) for i in range(0, 20, 2)]
