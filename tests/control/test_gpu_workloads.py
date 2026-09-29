"""Keep paired workload accounting isolated from cumulative server counters."""
import pytest

from atfm_experiments.gpu_cache import workloads


def test_request_measurement_subtracts_previous_cache_hits(monkeypatch, tmp_path):
    monkeypatch.setattr(workloads, 'metrics', lambda *args: 1000)
    monkeypatch.setattr(workloads, 'inference', lambda *args: {'external_hits': 1351, 'elapsed_s': .1})
    result = workloads.measured_inference(None, tmp_path, 'warm', range(352), 8)
    assert result['external_hits'] == 351


def test_matrix_alternates_pair_order(monkeypatch, tmp_path):
    monkeypatch.setattr(workloads, 'CASES', [workloads.Case('short', 64)])
    monkeypatch.setattr(workloads, 'run_case', lambda client, root, case, repeat, mode: (repeat, mode))
    assert workloads.run_cases(None, tmp_path, 2) == [(0, 'disk'), (0, 'prefetch'),
                                                    (1, 'prefetch'), (1, 'disk')]


@pytest.mark.parametrize('size', [0, 65, 2048])
def test_workload_rejects_invalid_prompt_size(monkeypatch, size):
    monkeypatch.setattr(workloads, 'request', lambda *args, **kwargs: {'tokens': list(range(4096))})
    with pytest.raises(ValueError):
        workloads.tokens_for(None, 'test', size)


def test_tokenization_preserves_requested_length_and_unique_prefix(monkeypatch):
    prompts = []
    def tokenize(*args, **kwargs):
        prompts.append(kwargs['json']['prompt'])
        return {'tokens': list(range(4096))}
    monkeypatch.setattr(workloads, 'request', tokenize)
    assert len(workloads.tokens_for(None, 'case-a', 1920)) == 1920
    assert len(workloads.tokens_for(None, 'case-b', 1920)) == 1920
    assert prompts[0][:24] != prompts[1][:24]
