"""Replica floor proposer (spec 6.4): q90 forecast demand at the scale-out lead time, divided by per-replica capacity."""
import numpy as np

from atfm.control import ReplicaDirective
from atfm.control.replica import ReplicaFloor, VirtualConnector
from atfm.schema.forecast import ForecastSnapshot


def _snap(horizons, kv_q, pf_q, n=32):
    H = len(horizons)
    z = np.zeros((H, n))
    kv = np.array(kv_q, float)[:, None] * np.ones((H, n))
    pf = np.array(pf_q, float)[:, None] * np.ones((H, n))
    return ForecastSnapshot(t=0.0, horizons=list(horizons), model_id="t",
                            samples={"kv_blocks": {"interactive": kv, "background": z}, "prefill_tokens": {"interactive": pf, "background": z}})


def test_floor_is_ceil_of_q90_demand_over_capacity_at_the_lead_time_horizon():
    snap = _snap([30.0, 300.0, 900.0], [100, 2500, 9000], [0, 0, 0])
    rf = ReplicaFloor(lead_time_s=300.0, blocks_per_replica=1000, prefill_tps_per_replica=1e9, min_replicas=1)
    d = rf.propose(now=0.0, snap=snap)
    assert isinstance(d, ReplicaDirective) and d.replicas_at_least == 3 and d.horizon_s == 300.0 and d.demand_q90_blocks == 2500.0
    assert d.expires_at == 300.0


def test_floor_respects_minimum_and_prefill_resource():
    snap = _snap([60.0], [10], [50000])                     # 50k prefill tokens over a 60 s horizon
    rf = ReplicaFloor(lead_time_s=60.0, blocks_per_replica=1000, prefill_tps_per_replica=400.0, min_replicas=2)
    assert rf.propose(0.0, snap).replicas_at_least == 3       # 50000 / (400 tok/s * 60 s) = 2.08 -> 3
    rf2 = ReplicaFloor(lead_time_s=60.0, blocks_per_replica=1000, prefill_tps_per_replica=1e9, min_replicas=2)
    assert rf2.propose(0.0, snap).replicas_at_least == 2      # floor


def test_virtual_connector_records_at_least_proposals():
    vc = VirtualConnector(current_replicas=1)
    snap = _snap([300.0], [2500], [0])
    rf = ReplicaFloor(lead_time_s=300.0, blocks_per_replica=1000, prefill_tps_per_replica=1e9)
    applied = vc.propose_at_least(rf.propose(0.0, snap))
    assert applied == 3 and vc.current_replicas == 3
    applied2 = vc.propose_at_least(ReplicaDirective(replicas_at_least=2, horizon_s=300.0, expires_at=600.0))
    assert applied2 == 3 and vc.current_replicas == 3         # AT_LEAST never scales down
    assert [h["replicas_at_least"] for h in vc.history] == [3, 2]
