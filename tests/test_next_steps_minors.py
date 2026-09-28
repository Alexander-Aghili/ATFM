"""Step 1: deployment-correctness minors from the review. Each test names the production change."""
import asyncio

import httpx
import numpy as np
import pytest

from atfm.control import ReplicaFloor, TierLogger


def test_tier_logger_and_replica_floor_validate_inputs():
    with pytest.raises(ValueError, match="tier"):
        TierLogger({})
    with pytest.raises(ValueError, match="lead_time_s"):
        ReplicaFloor(lead_time_s=0.0, blocks_per_replica=100, prefill_tps_per_replica=1.0)
    with pytest.raises(ValueError, match="blocks_per_replica"):
        ReplicaFloor(lead_time_s=10.0, blocks_per_replica=0, prefill_tps_per_replica=1.0)


async def test_directives_are_computed_once_per_snapshot_and_served_by_post():
    from atfm.board.forecaster import ExogenousModel, SessionForecaster
    from atfm.board.live import LiveBoard, SessionRegistry
    from atfm.board.predictors import SurvivalPredictor
    from atfm.board.service import create_board_app
    from atfm.control import TouchController
    from tests.board.test_service import _train
    train = _train()
    fc = SessionForecaster(SurvivalPredictor().fit(train), ExogenousModel().fit(train), horizons=[30.0, 120.0], n=16)
    board = LiveBoard(SessionRegistry(), fc)
    touch = TouchController(budget_per_s=1.0, tick_s=5.0)
    app = create_board_app(board, clock=lambda: 100.0, rng=np.random.default_rng(0), touch=touch)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://b") as c:
        await c.post("/tick")
        a = (await c.post("/directives")).json()
        b = (await c.post("/directives")).json()          # second poller within the same snapshot: same answer, no double spend
        assert a == b and touch.issued == 0 and a["snapshot_t"] == 100.0
        assert (await c.get("/directives")).status_code == 405


async def test_predict_enforces_the_time_budget_fail_open():
    from atfm.board.live import LiveBoard, SessionRegistry
    from atfm.board.service import create_board_app

    class SlowBoard(LiveBoard):
        def __init__(self):
            self.registry = SessionRegistry()
        def expected_tool_next(self, sid):
            import time; time.sleep(0.2); return 42.0
        def expected_service(self, sid, isl, osl):
            return 1.0
    app = create_board_app(SlowBoard(), budget_s=0.05)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://b") as c:
        p = (await c.post("/predict", json={"session_id": "x", "isl": 1, "osl": 1})).json()
    assert p["over_budget"] is True and p["e_tool_next_s"] == 0.0 and p["e_service_s"] == 0.0     # defaults, within budget


async def test_touches_are_counted_only_on_upstream_success():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from atfm.proxy.app import create_app
    from atfm.proxy.config import ProxyConfig
    up = _touch_upstream()
    app = create_app(ProxyConfig(upstream_url="http://up"), upstream_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=up), base_url="http://up"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://p") as c:
        await c.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
                     headers={"x-atfm-session": "s", "x-atfm-class": "background", "x-atfm-tenant": "t"})
        up.state.fail = True
        r = (await c.post("/touch", json={"session_id": "s"})).json()
        st = (await c.get("/state")).json()
        assert r["ok"] is False and st["touches"] == 0 and st["touch_failures"] == 1


def _touch_upstream():
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    up = FastAPI()
    up.state.fail = False

    @up.post("/v1/chat/completions")
    async def chat(req: Request):
        if up.state.fail:
            return JSONResponse({"error": "boom"}, status_code=503)
        return JSONResponse({"choices": [{"message": {"content": "k"}}], "usage": {"prompt_tokens": 9, "completion_tokens": 1}})
    return up
