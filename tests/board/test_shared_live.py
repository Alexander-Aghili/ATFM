"""Live and simulated forecasting use the same state and arrival contracts."""
from types import SimpleNamespace

import numpy as np
import pytest

from atfm.board.forecaster import ExogenousModel, SessionForecaster
from atfm.board.live import LiveBoard, SessionRegistry
from atfm.bus import InMemoryBus
from atfm.proxy.config import ProxyConfig
from atfm.schema.events import SessionStart
from atfm.sim.forecast_arm import ForecastPolicy


class Predictor:
    name = "M1_test"
    dm = SimpleNamespace(mean=lambda tool: {None: 5.0, "current": 10.0, "previous": 20.0}[tool])

    def resumption(self, state, now, n, rng):
        return rng.exponential(10.0, n)

    def next_call_isl(self, state, n, rng):
        return np.full(n, 32)

    def spawn(self, state, horizon, n, rng):
        return np.zeros(n, dtype=int)


def start(sid, t):
    return SessionStart(t=t, session_id=sid, tenant="tenant", cls="background")


def test_registry_reads_do_not_create_or_refresh_sessions():
    registry = SessionRegistry(expire_s=10.0)
    assert registry.get("missing") is None
    registry.apply(start("a", 0.0))
    ids = registry.session_ids()
    assert registry.get("a").session_id == "a"
    assert len(registry.states(10.0)) == 1
    assert registry.states(10.1) == []
    assert ids == ("a",)
    registry.apply(start("b", 20.0))
    for sid in registry.session_ids():
        registry.drop(sid)
    assert registry.states(20.0) == []


def test_arrival_windows_include_previous_tick_and_exclude_current_tick():
    registry = SessionRegistry()
    exo = ExogenousModel()
    board = LiveBoard(registry, SessionForecaster(Predictor(), exo, [10.0], n=4))
    for sid, t in (("a", 0.0), ("b", 5.0), ("c", 10.0)):
        registry.apply(start(sid, t))
    rng = np.random.default_rng(0)
    board.step(5.0, rng)
    assert exo.rate("background") == pytest.approx(1 / 60)
    board.step(5.0, rng)
    assert exo.rate("background") == pytest.approx(1 / 60)
    board.step(10.0, rng)
    assert exo.rate("background") == pytest.approx(2 / 60)
    board.step(11.0, rng)
    assert exo.rate("background") == pytest.approx(3 / 60)


@pytest.mark.parametrize("phase, tool, history, expected", [
    ("tool_running", "current", [("previous", 1.0)], 10.0),
    ("llm_pending", "current", [("previous", 1.0)], 20.0),
    ("llm_pending", None, [], 5.0),
])
def test_tool_lookup_uses_current_tool_then_history_then_pooled_mean(phase, tool, history, expected):
    registry = SessionRegistry()
    registry.apply(start("a", 0.0))
    state = registry.get("a")
    state.phase, state.tool_name, state.tool_history = phase, tool, history
    board = LiveBoard(registry, SessionForecaster(Predictor(), ExogenousModel(), [10.0]))
    assert board.expected_tool_next("a") == expected
    assert board.expected_tool_next("missing") == 5.0
    board.forecaster.predictor = SimpleNamespace()
    assert board.expected_tool_next("a") == 0.0


def test_live_and_simulated_forecasts_match_with_the_same_seed():
    predictor = Predictor()
    board = LiveBoard(SessionRegistry(), SessionForecaster(predictor, ExogenousModel(), [10.0, 30.0], n=8))
    policy = ForecastPolicy(4, ProxyConfig(upstream_url="x"), predictor, None, [10.0, 30.0], n=8)
    bus = InMemoryBus()
    sim = SimpleNamespace(events=bus, sessions={}, rng=np.random.default_rng(7))
    live_rng = np.random.default_rng(7)
    for t in (0.0, 5.0, 10.0):
        event = start(str(t), t)
        board.registry.apply(event)
        bus.publish(event)
        expected = board.step(t, live_rng)
        policy.on_tick(sim, t)
        for target, by_class in expected.samples.items():
            for cls, samples in by_class.items():
                np.testing.assert_array_equal(policy.snapshot.samples[target][cls], samples)
    assert sim.rng.random() == live_rng.random()
    sim.sessions["0.0"] = SimpleNamespace(done=True)
    policy.on_tick(sim, 11.0)
    assert policy.registry.get("0.0") is None
