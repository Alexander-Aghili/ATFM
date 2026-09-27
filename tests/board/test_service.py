"""Board HTTP service: snapshot quantiles, per-request predictions under a time budget, directives feed."""
import numpy as np
import httpx
import pytest

from atfm.board.forecaster import ExogenousModel, SessionForecaster
from atfm.board.live import LiveBoard, SessionRegistry
from atfm.board.predictors import SurvivalPredictor
from atfm.board.service import create_board_app
from atfm.bus import InMemoryBus
from atfm.schema.events import LlmRequest, SessionStart, ToolStart
from atfm.schema.trace import TraceRow, TraceTable


def _train():
    rows = []
    for k in range(20):
        t = 1000.0 * k
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=0, t_request=t, t_first_token=t + 1,
                             t_last_token=t + 2, isl=100, osl=10, tool_name="pytest", backend_id="ci", t_tool_start=t + 2,
                             t_tool_end=t + 62 + k, source="test"))
        rows.append(TraceRow(session_id=f"s{k}", cls="background", tenant="t", turn_index=1, t_request=t + 70,
                             t_first_token=t + 71, t_last_token=t + 72, isl=150, osl=10, tool_name=None, source="test"))
    return TraceTable.from_rows(rows)


@pytest.fixture
def board():
    train = _train()
    fc = SessionForecaster(SurvivalPredictor().fit(train), ExogenousModel().fit(train), horizons=[30.0, 120.0], n=32)
    return LiveBoard(SessionRegistry(), fc, tick_s=5.0)


async def test_snapshot_predict_and_directives(board):
    bus = InMemoryBus()
    clock = {"t": 100.0}
    app = create_board_app(board, bus=bus, clock=lambda: clock["t"], rng=np.random.default_rng(0), budget_s=0.05)
    bus.publish(SessionStart(t=90.0, session_id="x", tenant="t", cls="background"))
    bus.publish(LlmRequest(t=91.0, session_id="x", turn_index=0, request_id="r", isl=100))
    bus.publish(ToolStart(t=95.0, session_id="x", turn_index=0, call_id="c", tool_name="pytest", backend_id="ci"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://board") as c:
        r = await c.post("/tick")
        assert r.status_code == 200 and r.json()["sessions"] == 1
        s = (await c.get("/snapshot")).json()
        assert s["horizons"] == [30.0, 120.0] and set(s["q90"]) == {"kv_blocks", "prefill_tokens"}
        assert len(s["q90"]["kv_blocks"]["background"]) == 2 and s["t"] == 100.0
        p = (await c.post("/predict", json={"session_id": "x", "isl": 120, "osl": 20})).json()
        assert p["e_service_s"] > 0 and 30.0 < p["e_tool_next_s"] < 120.0 and p["elapsed_ms"] >= 0
        p2 = (await c.post("/predict", json={"session_id": "unknown", "isl": 120, "osl": 20})).json()
        assert p2["e_tool_next_s"] > 0                                   # pooled fallback, never an error
        d = (await c.post("/directives")).json()
        assert d["holds"] == [] and d["touches"] == [] and d["tier"] == [] and "replica" in d
        assert (await c.get("/healthz")).json()["ok"] is True
