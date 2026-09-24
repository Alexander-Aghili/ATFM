from atfm.proxy.config import ProxyConfig
from atfm.proxy.index import CallMeta, estimate_isl, service_time, compute_index, tier, priority_bucket

def _meta(cls="background", deadline=None, isl=2000, osl=100):
    return CallMeta(session_id="s", cls=cls, tenant="t", deadline=deadline, parent=None, turn_index=0,
                    isl=isl, predicted_osl=osl, t_arrival=1000.0)

def test_estimate_isl_from_messages():
    assert estimate_isl({"messages": [{"role": "user", "content": "a" * 400}, {"role": "system", "content": "b" * 40}]}) == 110
    assert estimate_isl({"messages": []}) == 1

def test_service_time_and_index():
    cfg = ProxyConfig(upstream_url="http://u", beta=0.5)
    m = _meta(isl=20000, osl=60)
    assert abs(service_time(m, cfg) - 2.0) < 1e-9
    assert abs(compute_index(m, cfg, e_service_s=2.0, e_tool_next_s=4.0) - 1.0 * 3.0 / 2.0) < 1e-9
    assert compute_index(_meta("interactive", isl=20000, osl=60), cfg, 2.0, 0.0) == 5.0

def test_tiers():
    cfg = ProxyConfig(upstream_url="http://u", slack_threshold_s=5.0)
    assert tier(_meta("background"), cfg, now=1000.0, e_service_s=1.0) == 0
    assert tier(_meta("interactive"), cfg, now=1000.0, e_service_s=1.0) == 1
    assert tier(_meta("interactive", deadline=1004.0), cfg, now=1000.0, e_service_s=1.0) == 2

def test_priority_bucket_quartiles():
    assert priority_bucket(10.0, [1.0, 2.0, 3.0, 10.0]) == 3
    assert priority_bucket(1.0, [1.0, 2.0, 3.0, 10.0]) == 0
    assert priority_bucket(5.0, []) == 3
