"""Step 2: the deploy path wired end to end — proxy asks the board over HTTP, the board attaches
controllers and a metrics scrape from config, residency for touches comes from the registry."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import numpy as np
import pytest

from atfm.board.forecaster import ExogenousModel, SessionForecaster
from atfm.board.live import LiveBoard, SessionRegistry
from atfm.board.predictors import SurvivalPredictor
from atfm.board.service import configure_controllers, create_board_app, residency_from_registry
from atfm.schema.events import LlmDone, LlmRequest, SessionStart, ToolEnd, ToolStart
from tests.board.test_service import _train


class _Board(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0)); body = json.loads(self.rfile.read(n))
        self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"e_service_s": 0.5, "e_tool_next_s": 77.0 if body["session_id"] == "known" else 5.0,
                                     "elapsed_ms": 1.0, "over_budget": False}).encode())
    def log_message(self, *a): pass


@pytest.fixture
def board_server():
    srv = HTTPServer(("127.0.0.1", 0), _Board)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_board_client_predictor_speaks_the_board_protocol(board_server):
    from atfm.proxy.board_client import BoardClient
    bc = BoardClient(board_server, timeout_s=1.0)
    assert bc.expected_tool_next("known") == 77.0 and bc.expected_service("known", 100, 10) == 0.5
    dead = BoardClient("http://127.0.0.1:1", timeout_s=0.2)
    assert dead.expected_tool_next("known") == 0.0 and dead.expected_service("k", 1, 1) == 0.0      # fail-open


async def test_proxy_uses_the_board_when_board_url_is_configured(board_server):
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    from atfm.proxy.app import create_app
    from atfm.proxy.config import ProxyConfig
    up = FastAPI(); up.state.seen = []

    @up.post("/v1/chat/completions")
    async def chat(req: Request):
        up.state.seen.append(await req.json())
        return JSONResponse({"choices": [{"message": {"content": "k"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    app = create_app(ProxyConfig(upstream_url="http://up", board_url=board_server, beta=1.0, window=4),
                     upstream_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=up), base_url="http://up"))
    assert app.state.predictor is not None
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://p") as c:
        r = await c.post("/v1/chat/completions", json={"model": "m", "messages": [{"role": "user", "content": "x"}]},
                         headers={"x-atfm-session": "known", "x-atfm-class": "background", "x-atfm-tenant": "t"})
        assert r.status_code == 200
    assert app.state.last_index["known"] == pytest.approx(1.0 * (1.0 + 1.0 * 77.0) / 0.5)          # index used the board's numbers


def test_configure_controllers_from_config_and_scrape_feeds_capacity_and_frontier():
    train = _train()
    fc = SessionForecaster(SurvivalPredictor().fit(train), ExogenousModel().fit(train), horizons=[30.0, 120.0], n=16)
    app = create_board_app(LiveBoard(SessionRegistry(), fc), clock=lambda: 100.0, rng=np.random.default_rng(0))
    cfg = {"gdp": {"slot_s": 30.0, "horizon_s": 120.0, "eps": 0.1, "max_hold_s": 60.0},
           "touch": {"horizon_s": 30.0, "budget_per_s": 1.0}, "tier": {"tier_lead_s": {"gpu": 0.0, "cpu": 5.0}},
           "replica": {"lead_time_s": 120.0, "blocks_per_replica": 8000, "prefill_tps_per_replica": 20000.0},
           "metrics": {"default_total_blocks": 8000, "default_worker_id": "w0", "pressure_threshold": 0.9}}
    page = ['vllm:gpu_cache_usage_perc{model_name="m"} 0.95\nvllm:num_requests_waiting{model_name="m"} 2\n']
    scrape = configure_controllers(app, cfg, fetch=lambda: page[0])
    st = app.state
    assert st.gdp is not None and st.touch is not None and st.tier is not None and st.replica is not None
    scrape()
    assert st.capacity == {"kv_blocks": 400.0, "prefill_tokens": 20000.0 * 30.0}                    # free blocks, prefill per slot
    assert st.frontier == {"w0": 0.0} and st.worker_metrics["w0"].kv_blocks_used == 7600               # under pressure: all at risk
    page[0] = 'vllm:gpu_cache_usage_perc{model_name="m"} 0.5\n'
    scrape()
    assert st.frontier == {} and st.capacity["kv_blocks"] == 4000.0                                     # no pressure: nothing at risk


def test_residency_from_registry_uses_last_call_time_and_context_size():
    reg = SessionRegistry()
    for e in [SessionStart(t=0.0, session_id="a", tenant="t", cls="background"),
              LlmRequest(t=1.0, session_id="a", turn_index=0, request_id="r1", isl=1600),
              LlmDone(t=3.0, session_id="a", request_id="r1", osl=32, worker_id="w1"),
              ToolStart(t=3.0, session_id="a", turn_index=0, call_id="c", tool_name="pytest"),
              SessionStart(t=0.0, session_id="b", tenant="t", cls="interactive"),
              LlmRequest(t=5.0, session_id="b", turn_index=0, request_id="r2", isl=800)]:
        reg.apply(e)
    res = residency_from_registry(reg, now=10.0, block_size=16, default_worker="w0")
    assert res["a"].worker_id == "w1" and res["a"].blocks == 1600 // 16 + 1 and res["a"].last_used == 3.0   # registry ctx = last isl
    assert "b" not in res                                                                            # still running its call
