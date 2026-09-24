import numpy as np
from atfm.schema.trace import TraceRow, TraceTable
from atfm.board.predictors import SurvivalPredictor, ProgressPredictor
from atfm.eval.resumption import resumption_records, score_resumption

def _table(n=12, dur=100.0, with_progress=True):
    rows = []
    for k in range(n):
        t = k * 500.0
        step = dur / 10.0   # progress paced to the tool's actual duration
        prog = [{"t": t + 2 + step * j, "completed": 10 * j, "total": 100, "phase": "run"} for j in range(1, 10)] if with_progress else []
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=0, t_request=t, t_first_token=t + 1,
                             t_last_token=t + 2, isl=100, osl=10, tool_name="pytest", backend_id="ci", t_tool_start=t + 2,
                             t_tool_end=t + 2 + dur, progress_events=prog, source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=1, t_request=t + 2 + dur,
                             t_first_token=t + 3 + dur, t_last_token=t + 4 + dur, isl=150, osl=10, tool_name=None, source="test"))
    return TraceTable.from_rows(rows)

def test_resumption_records_observe_every_tool_at_fixed_offsets():
    recs = resumption_records(_table(n=3), offsets_s=[10.0, 50.0])
    assert len(recs) == 6 and {r["tool_name"] for r in recs} == {"pytest"}
    r = [x for x in recs if x["elapsed"] == 50.0][0]
    assert abs(r["true_remaining"] - 50.0) < 1e-9 and r["signal"] == "strong" and r["state"].phase == "tool_running"
    assert len(r["state"].progress) == 5                      # only events up to the observation time (50 s of a 100 s tool)

def test_score_resumption_progress_beats_survival_when_rate_differs():
    # test tools run at half the training rate (200 s instead of 100 s): progress sees it, elapsed time alone cannot
    train = _table(n=20, dur=100.0)
    test = _table(n=6, dur=200.0)
    m1 = SurvivalPredictor().fit(train); m2 = ProgressPredictor().fit(train)
    recs = resumption_records(test, offsets_s=[40.0, 80.0, 120.0])
    df = score_resumption({"M1": m1, "M2": m2}, recs, n=400, rng=np.random.default_rng(0))
    assert set(df.columns) >= {"model", "tool_name", "signal", "elapsed", "crps", "pinball90", "n"}
    pin = df.groupby("model")["pinball90"].mean()
    assert pin["M2"] < 0.5 * pin["M1"]
