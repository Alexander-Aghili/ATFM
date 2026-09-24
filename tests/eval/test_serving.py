import numpy as np, pandas as pd
from atfm.eval.serving import serving_metrics, paired_bootstrap

def _log():
    return pd.DataFrame([
        {"session_id": "i1", "class": "interactive", "tenant": "a", "turn_index": 1, "t_arrival": 10.0, "t_release": 10.0, "t_first_token": 11.0, "t_end": 12.0, "isl": 1000, "prefix_hit_tokens": 800, "recomputed_tokens": 200, "held_s": 0.0, "queue_proxy_s": 0.0, "queue_worker_s": 0.5, "hold_kv_block_s": 0.0, "evictions_caused": 0, "deadline_missed": False},
        {"session_id": "i1", "class": "interactive", "tenant": "a", "turn_index": 2, "t_arrival": 20.0, "t_release": 20.0, "t_first_token": 23.0, "t_end": 24.0, "isl": 1200, "prefix_hit_tokens": 0, "recomputed_tokens": 1200, "held_s": 0.0, "queue_proxy_s": 0.0, "queue_worker_s": 2.5, "hold_kv_block_s": 0.0, "evictions_caused": 0, "deadline_missed": False},
        {"session_id": "b1", "class": "background", "tenant": "b", "turn_index": 1, "t_arrival": 5.0, "t_release": 35.0, "t_first_token": 36.0, "t_end": 40.0, "isl": 2000, "prefix_hit_tokens": 1000, "recomputed_tokens": 1000, "held_s": 30.0, "queue_proxy_s": 30.0, "queue_worker_s": 0.0, "hold_kv_block_s": 3000.0, "evictions_caused": 2, "deadline_missed": True},
    ])

def test_serving_metrics_values():
    sessions = [{"session_id": "i1", "class": "interactive", "tenant": "a", "t_start": 0.0, "t_end": 24.0, "deadline": 100.0, "missed": False, "turns": 3},
                {"session_id": "b1", "class": "background", "tenant": "b", "t_start": 0.0, "t_end": 40.0, "deadline": 30.0, "missed": True, "turns": 2}]
    m = serving_metrics(_log(), sessions, slo_ttft_s=2.0, sim_duration_s=3600.0, gpu_count=1)
    assert m["ttft_after_tool_p50"] == 2.0 and m["slo_attainment"] == 0.5
    assert m["bg_jct_mean"] == 40.0 and m["deadline_hit_rate"] == 0.5 and m["tasks_per_hour"] == 2.0
    assert m["max_imposed_delay_by_tenant"] == {"a": 0.0, "b": 30.0} and m["gpu_hours"] == 1.0
    assert m["recomputed_prefill_tokens"] == 2400 and abs(m["kv_hit_rate"] - 1800 / 4200) < 1e-9
    assert m["hold_kv_block_s"] == 3000.0 and m["evictions_caused_by_holds"] == 2
    assert abs(m["queue_proxy_share"] - 30.0 / 33.0) < 1e-9

def test_paired_bootstrap_same_sessions_across_arms():
    rng = np.random.default_rng(0)
    base = pd.DataFrame({"session_id": [f"s{i}" for i in range(50)], "value": rng.normal(10, 1, 50)})
    better = base.assign(value=base.value - 1.0)
    out = paired_bootstrap({"A": base, "B": better}, metric_fn=lambda df: df.value.mean(), n_boot=200, rng=np.random.default_rng(1))
    b = out[out.arm == "B"].iloc[0]
    assert abs(b["diff_mean"] + 1.0) < 0.05 and b["diff_ci_hi"] < 0.0
