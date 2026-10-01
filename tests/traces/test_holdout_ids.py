"""Every replayed root, including each root of a fleet case, is held out of board training."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('build_holdout_train', Path(__file__).parents[2] / 'scripts/build_holdout_train.py')
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_held_out_ids_cover_single_roots_and_fleets():
    selection = {'cases': {'one': {'trace_id': 'a'}, 'fleet': {'trace_ids': ['b', 'c']}}}
    assert build.held_out_ids(selection) == ['a', 'b', 'c']
