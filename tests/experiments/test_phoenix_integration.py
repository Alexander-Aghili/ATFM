"""Opt-in actual Phoenix/SQLite/OTLP round trip, without credentials or an LLM."""
import os

import pytest

pytest.importorskip("phoenix.client")
pytest.importorskip("opentelemetry.sdk")


@pytest.mark.skipif(os.environ.get("ATFM_TEST_PHOENIX") != "1", reason="set ATFM_TEST_PHOENIX=1 to start a local Phoenix server")
def test_local_phoenix_round_trip(tmp_path):
    from atfm_experiments.phoenix_smoke import local_phoenix, smoke

    with local_phoenix(tmp_path / "server") as endpoint:
        summary = smoke(endpoint, tmp_path / "results")
    assert summary["tools"] == 2
    assert summary["failures"] == 1
    assert summary["progress_events"] == 6
    assert summary["persisted_spans"] == 3
    assert summary["verified_annotations"] == 2
