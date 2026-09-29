"""Evidence checks must reject a working HTTP endpoint without external reuse."""
import pytest

from atfm_experiments.gpu_cache.probe import metric, validate
from atfm_experiments.gpu_cache.stack import commands, REVISION


def test_metric_ignores_help_and_sums_engine_labels():
    text = '# HELP hits description\nhits{engine="0"} 351.0\nhits{engine="1"} 12\n'
    assert metric(text, 'hits') == 363
    with pytest.raises(ValueError, match='missing metric'):
        metric(text, 'queries')


@pytest.mark.parametrize('cold,warm', [(1, 352), (0, 0), (0, 350)])
def test_probe_rejects_preexisting_cache_or_insufficient_reuse(cold, warm):
    with pytest.raises(RuntimeError, match='reuse not proven'):
        validate(dict(external_hits=cold, text='same'), dict(external_hits=warm, text='same'), range(352))


def test_probe_rejects_changed_output():
    with pytest.raises(RuntimeError, match='output changed'):
        validate(dict(external_hits=0, text='a'), dict(external_hits=351, text='b'), range(352))


def test_launch_pins_model_and_disables_gpu_prefix_cache(tmp_path):
    cache, engine = commands(tmp_path / 'venv', tmp_path / 'out')
    assert engine.count(REVISION) == 2
    assert '--no-enable-prefix-caching' in engine
    assert '--host' in cache and '127.0.0.1' in cache


def test_runner_refuses_an_occupied_port(monkeypatch):
    import socket
    from atfm_experiments.gpu_cache import stack
    with socket.socket() as occupied:
        occupied.bind(('127.0.0.1', 0))
        monkeypatch.setattr(stack, 'PORTS', (occupied.getsockname()[1],))
        with pytest.raises(OSError):
            stack.check_ports()


def test_failed_startup_cleans_up_owned_process(tmp_path, monkeypatch):
    import sys
    from atfm_experiments.gpu_cache import stack
    processes = []
    def fail_ready(process, url):
        processes.append(process)
        raise RuntimeError('test startup failure')
    monkeypatch.setattr(stack, 'ready', fail_ready)
    with pytest.raises(RuntimeError, match='test startup failure'):
        with stack.server([sys.executable, '-c', 'import time; time.sleep(30)'], tmp_path, 'failed', 'unused'):
            pytest.fail('startup must not succeed')
    assert len(processes) == 1 and processes[0].poll() is not None
