import hashlib
import json

import pytest

from atfm_experiments.benchmark_gdp import digest, workload


@pytest.mark.parametrize("saturated,mixed", [(False, False), (True, False), (False, True)])
def test_workload_is_repeatable_and_streaming_digest_matches_json(saturated, mixed):
    run = workload(20, 8, 7, saturated, mixed)
    result = run()
    assert len(result) == 20
    assert result == run()
    expected = hashlib.sha256(json.dumps([d.__dict__ for d in result], sort_keys=True).encode()).hexdigest()
    assert digest(result) == expected


def test_empty_digest_matches_json():
    assert digest([]) == hashlib.sha256(b"[]").hexdigest()
