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
