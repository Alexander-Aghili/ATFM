import copy

import numpy as np
import pytest

from atfm.board.sampling import empirical_draw


@pytest.mark.parametrize('generator', [np.random.PCG64, np.random.Philox, np.random.MT19937, np.random.SFC64])
@pytest.mark.parametrize('n', [0, 1, 7, 128])
@pytest.mark.parametrize('values', [np.array([1]), np.array([1., np.inf, 4.]), np.arange(33)[::2]])
def test_empirical_draw_matches_choice_and_rng_state(generator, n, values):
    original = np.random.Generator(generator(731))
    updated = copy.deepcopy(original)
    np.testing.assert_array_equal(empirical_draw(values, n, updated), original.choice(values, size=n, replace=True))
    np.testing.assert_array_equal(updated.random(100), original.random(100))


def test_empty_empirical_distribution_matches_choice():
    values = np.array([])
    assert empirical_draw(values, 0, np.random.default_rng(0)).size == 0
    with pytest.raises(ValueError):
        empirical_draw(values, 1, np.random.default_rng(0))


def test_forecast_preserves_samples_and_rng_with_legacy_sampling(monkeypatch, tmp_path):
    from atfm_experiments.benchmark_cpu import make_case
    from atfm.board import forecaster
    from atfm.board.predictors import duration, progress
    run = make_case('forecast', 25, tmp_path)
    expected = run()
    for module in (forecaster, duration, progress):
        monkeypatch.setattr(module, 'empirical_draw', lambda values, n, rng: rng.choice(values, size=n, replace=True))
    np.testing.assert_array_equal(run(), expected)
