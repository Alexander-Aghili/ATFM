"""Every run directory records its provenance (spec 9): git SHA, config hash, inputs hash, package version."""
import hashlib
import json

import numpy as np
import pandas as pd

from atfm.experiments.h2sim import H2SimConfig, run_h2sim, write_manifest
from atfm.schema.trace import TraceRow, TraceTable


def test_manifest_records_git_sha_config_hash_and_inputs(tmp_path):
    cfg = H2SimConfig(name="prov", regime="short_tool", duration_s=60.0, seeds=[0], arms=["native"], out_dir=str(tmp_path))
    m = write_manifest(tmp_path / "prov", cfg, inputs=[])
    assert len(m["git_sha"]) >= 7 and m["config_sha256"] == hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()
    assert m["atfm_version"] and m["inputs"] == [] and "created_at" in m
    assert json.loads((tmp_path / "prov" / "manifest.json").read_text())["git_sha"] == m["git_sha"]


def test_trace_regime_replays_a_parquet_table_and_records_its_hash(tmp_path):
    rows = []
    for k in range(6):
        t = 100.0 * k
        rows.append(TraceRow(session_id=f"s{k}", cls="interactive" if k % 2 else "background", tenant="t", turn_index=0,
                             t_request=t, t_first_token=t + 1, t_last_token=t + 2, isl=1200, osl=20, tool_name="bash", backend_id="local",
                             t_tool_start=t + 2, t_tool_end=t + 5, source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="interactive" if k % 2 else "background", tenant="t", turn_index=1,
                             t_request=t + 6, t_first_token=t + 7, t_last_token=t + 8, isl=1300, osl=20, tool_name=None, source="test"))
    path = tmp_path / "t.parquet"
    TraceTable.from_rows(rows).to_parquet(path)
    cfg = H2SimConfig(name="trace", regime="trace", trace_path=str(path), duration_s=600.0, seeds=[0], arms=["native", "proxy_rules"],
                      out_dir=str(tmp_path), trace_rate_per_hour=120.0)
    df = run_h2sim(cfg)
    assert set(df["arm"]) == {"native", "proxy_rules"} and (df["sessions_completed"] > 0).all()
    m = json.loads((tmp_path / "trace" / "manifest.json").read_text())
    assert m["inputs"][0]["path"].endswith("t.parquet") and len(m["inputs"][0]["sha256"]) == 64
