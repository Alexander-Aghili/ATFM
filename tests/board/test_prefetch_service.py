"""The board serves forecast-driven prefetch directives alongside its other tier directives."""
import httpx
import numpy as np

from atfm.board.service import configure_controllers, create_board_app
from atfm.bus import InMemoryBus
from atfm.control.prefetch import PrefetchPlanner
from atfm.schema.events import LlmRequest, SessionStart, ToolStart
from tests.board.test_service import board  # noqa: F401  (fixture)

PREFETCH = {"bytes_per_token": 147456, "warm_bytes_per_s": 1.4e9, "overhead_s": 0.2, "margin_s": 1.0,
            "budget_bytes": 2**30, "max_per_plan": 4}


async def directives(board, interval_s):
    bus = InMemoryBus()
    app = create_board_app(board, bus=bus, clock=lambda: 100.0, rng=np.random.default_rng(0), budget_s=0.05)
    configure_controllers(app, {"prefetch": dict(PREFETCH, interval_s=interval_s)})
    assert isinstance(app.state.prefetch, PrefetchPlanner)
    bus.publish(SessionStart(t=90.0, session_id="x", tenant="t", cls="background"))
    bus.publish(LlmRequest(t=91.0, session_id="x", turn_index=0, request_id="r", isl=4096))
    bus.publish(ToolStart(t=95.0, session_id="x", turn_index=0, call_id="c", tool_name="pytest", backend_id="ci"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://board") as c:
        await c.post("/tick")
        return (await c.post("/directives")).json()["tier"]


async def test_prefetch_directive_when_return_is_within_the_window(board):  # noqa: F811
    (d,) = await directives(board, interval_s=120.0)
    assert (d["session_id"], d["action"], d["tier"]) == ("x", "prefetch", "cpu")


async def test_no_prefetch_while_the_return_is_far_away(board):  # noqa: F811
    assert await directives(board, interval_s=1.0) == []
