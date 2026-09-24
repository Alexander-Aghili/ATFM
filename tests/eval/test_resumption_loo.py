import numpy as np

from atfm.eval.resumption import job_family, leave_one_family_out
from atfm.schema.trace import TraceRow, TraceTable


def _row(sid, t0, dur):
    return TraceRow(session_id=sid, cls="background", tenant="t", turn_index=0, t_request=t0, t_first_token=t0 + 1,
                    t_last_token=t0 + 2, isl=100, osl=10, tool_name="pipe", backend_id="local", t_tool_start=t0 + 2,
                    t_tool_end=t0 + 2 + dur, progress_events=[{"t": t0 + 2 + dur / 2, "completed": 50, "total": 100, "phase": "run"}],
                    source="test")


def test_job_family_strips_repeat_index_and_hash():
    assert job_family("pipe-s-a-0-27f780") == "pipe-s-a"
    assert job_family("build-fmt-j2-1-c15914") == "build-fmt-j2"
    assert job_family("weird") == "weird"


def test_leave_one_family_out_scores_each_family_against_models_fit_on_the_rest():
    rows = [_row(f"pipe-a-{k}-{k:06x}", 1000.0 * k, 100.0 + 5 * k) for k in range(3)] + \
           [_row(f"pipe-b-{k}-{k:06x}", 5000.0 + 1000.0 * k, 200.0 + 5 * k) for k in range(3)]
    table = TraceTable.from_rows(rows)
    df = leave_one_family_out(table, models=["B2", "M1"], offsets_s=[15.0, 45.0], min_duration_s=30.0,
                              n=64, rng=np.random.default_rng(0))
    assert set(df["family"]) == {"pipe-a", "pipe-b"}
    assert set(df["model"]) == {"B2", "M1"}
    # every phase scored at both offsets for each model
    assert (df.groupby(["model", "family"])["n"].sum() == 6).all()
    assert np.isfinite(df["pinball90"]).all() and (df["pinball90"] >= 0).all()
