"""Queue-aware placement: rank by time to the next KV *use* (predicted arrival plus the class's recent proxy
queue wait), not by arrival. Each test names the production change that makes it pass."""
import numpy as np
import pandas as pd

from atfm.proxy.config import ProxyConfig
from atfm.sim.engine import EngineConfig


def test_recent_queue_wait_by_class_from_the_call_log():
    from atfm.sim.kv_placement import recent_queue_wait
    rows = [{"class": "background", "t_release": 100.0, "queue_proxy_s": 300.0}, {"class": "background", "t_release": 150.0, "queue_proxy_s": 500.0},
            {"class": "interactive", "t_release": 160.0, "queue_proxy_s": 1.0}, {"class": "background", "t_release": 10.0, "queue_proxy_s": 9999.0}]
    q = recent_queue_wait(rows, now=170.0, window_s=100.0)
    assert q == {"background": 400.0, "interactive": 1.0}            # the 10.0 row is outside the window
    assert recent_queue_wait([], now=0.0, window_s=100.0) == {}


def test_queue_aware_order_adds_the_class_wait_to_absence():
    from atfm.sim.kv_placement import order_victims
    eta = {"bg": 20.0, "it": 60.0}
    cls = {"bg": "background", "it": "interactive"}
    assert order_victims(["bg", "it"], eta, {}, now=0.0, size_aware=False, cls=cls)[0] == "it"          # by arrival: interactive later
    q = {"background": 500.0, "interactive": 1.0}
    assert order_victims(["bg", "it"], eta, {}, now=0.0, size_aware=False, cls=cls, queue_wait=q)[0] == "bg"   # by next use: background later


def test_queue_aware_arms_registered_and_use_the_simulators_recent_queue_times():
    from atfm_experiments.h2sim import ARMS, H2SimConfig, _arm
    from atfm.sim.core import Simulator
    from atfm.sim.kv_placement import OracleKvPolicy
    from tests.sim.test_touch_arm import _prog
    assert {"forecast_M1_kv_q", "forecast_M2_kv_q", "oracle_kv_q"} <= set(ARMS)
    cfg = H2SimConfig(name="t", regime="short_tool", queue_window_s=120.0)
    engines = [EngineConfig(**e) for e in cfg.engines]
    pol = _arm("oracle_kv_q", cfg, engines, None, np.random.default_rng(0))
    assert isinstance(pol, OracleKvPolicy) and pol.queue_aware and pol.name == "oracle_kv_q"
    progs = [_prog("a", "background", 0.0, 800, "bash", 5.0, turns=3), _prog("b", "interactive", 0.0, 800, "bash", 5.0, turns=3)]
    sim = Simulator(progs, engines, pol, rng=np.random.default_rng(0))
    sim.run()
    assert set(pol.last_queue_wait) <= {"background", "interactive"} and all(v >= 0 for v in pol.last_queue_wait.values())
