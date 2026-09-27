"""Contracts for placement summaries shared by the service and simulator."""
from types import SimpleNamespace

import numpy as np
import pytest

from atfm.board.resumption import resumption_quantiles


def states(*ids):
    return [SimpleNamespace(session_id=sid) for sid in ids]


@pytest.mark.parametrize("samples, expected", [
    ([0.0, 10.0, 20.0], (2.0, 10.0, 18.0)),
    ([np.nan, np.inf, -np.inf, 10.0], (10.0, 10.0, 10.0)),
    ([np.nan, np.inf], None),
    ([], None),
])
def test_quantiles_are_conditional_on_finite_draws(samples, expected):
    predictor = SimpleNamespace(resumption=lambda *args: np.array(samples))
    result = resumption_quantiles(predictor, states("a"), 100.0, 4, np.random.default_rng(0))
    assert result == ({} if expected is None else {"a": expected})
    if expected is not None:
        assert all(type(value) is float for value in result["a"])


def test_errors_propagate_by_default_but_serving_can_isolate_a_session():
    def draw(state, now, n, rng):
        if state.session_id == "broken":
            raise ValueError("invalid model state")
        return np.array([1.0, 2.0, 3.0])

    predictor = SimpleNamespace(resumption=draw)
    inputs = states("before", "broken", "after")
    with pytest.raises(ValueError, match="invalid model state"):
        resumption_quantiles(predictor, inputs, 100.0, 3, np.random.default_rng(0))
    result = resumption_quantiles(predictor, inputs, 100.0, 3, np.random.default_rng(0), skip_errors=True)
    assert result == {"before": (1.2, 2.0, 2.8), "after": (1.2, 2.0, 2.8)}


def test_random_draw_order_matches_direct_predictor_calls():
    calls = []

    def draw(state, now, n, rng):
        calls.append((state.session_id, now, n))
        return rng.exponential(10.0, n)

    actual_rng, expected_rng = np.random.default_rng(42), np.random.default_rng(42)
    result = resumption_quantiles(SimpleNamespace(resumption=draw), iter(states("b", "a")), 7.0, 16, actual_rng)
    expected = {sid: tuple(np.quantile(expected_rng.exponential(10.0, 16), [0.1, 0.5, 0.9])) for sid in ("b", "a")}
    assert result == expected
    assert calls == [("b", 7.0, 16), ("a", 7.0, 16)]
    assert actual_rng.random() == expected_rng.random()


def test_empty_fleet_does_not_draw():
    def draw(*args):
        pytest.fail("an empty fleet must not call the predictor")

    assert resumption_quantiles(SimpleNamespace(resumption=draw), [], 0.0, 16, np.random.default_rng(0)) == {}
