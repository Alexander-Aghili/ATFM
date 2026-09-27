"""Prometheus text -> WorkerMetrics events (Dynamo / vLLM worker metrics, spec 5.1 inputs)."""
from atfm.board.metrics import parse_prometheus, worker_metrics_from_prometheus
from atfm.schema.events import WorkerMetrics

SAMPLE = """# HELP vllm:gpu_cache_usage_perc GPU KV-cache usage. 1 means 100 percent usage.
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc{model_name="m",worker="w0"} 0.25
vllm:num_requests_waiting{model_name="m",worker="w0"} 3
vllm:num_requests_running{model_name="m",worker="w0"} 5
dynamo_component_kv_total_blocks{component="backend",instance="w0"} 8000
dynamo_component_kv_active_blocks{component="backend",instance="w0"} 2000
dynamo_component_kv_total_blocks{component="backend",instance="w1"} 8000
dynamo_component_kv_active_blocks{component="backend",instance="w1"} 6400
dynamo_component_requests_pending{component="backend",instance="w1"} 7
"""


def test_parse_prometheus_text_keeps_labels_and_values():
    rows = parse_prometheus(SAMPLE)
    assert ("vllm:gpu_cache_usage_perc", {"model_name": "m", "worker": "w0"}, 0.25) in rows
    assert ("dynamo_component_kv_active_blocks", {"component": "backend", "instance": "w1"}, 6400.0) in rows
    assert all(not name.startswith("#") for name, _, _ in rows)


def test_worker_metrics_events_one_per_worker_with_blocks_and_queue_depth():
    evs = worker_metrics_from_prometheus(SAMPLE, t=12.0)
    by = {e.worker_id: e for e in evs}
    assert set(by) == {"w0", "w1"} and all(isinstance(e, WorkerMetrics) and e.t == 12.0 for e in evs)
    assert by["w0"].kv_blocks_total == 8000 and by["w0"].kv_blocks_used == 2000 and by["w0"].queue_depth == 3
    assert by["w1"].kv_blocks_used == 6400 and by["w1"].queue_depth == 7


def test_vllm_only_page_falls_back_to_usage_fraction_and_a_configured_total():
    page = 'vllm:gpu_cache_usage_perc{worker="a"} 0.5\nvllm:num_requests_waiting{worker="a"} 2\n'
    evs = worker_metrics_from_prometheus(page, t=1.0, default_total_blocks=1000)
    assert len(evs) == 1 and evs[0].worker_id == "a" and evs[0].kv_blocks_used == 500 and evs[0].kv_blocks_total == 1000
    assert worker_metrics_from_prometheus("garbage\n", t=1.0) == []
