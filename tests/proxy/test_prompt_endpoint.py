"""The proxy hands the control loop a session's last prompt so the LMCache actuator can tokenize it."""
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from atfm.proxy.app import create_app
from atfm.proxy.config import ProxyConfig


async def test_session_prompt_endpoint_returns_messages_and_model():
    up = FastAPI()

    @up.post("/v1/chat/completions")
    async def chat(req: Request):
        return JSONResponse({"choices": [{"message": {"content": "k"}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})
    app = create_app(ProxyConfig(upstream_url="http://up"), upstream_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=up), base_url="http://up"))
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://p") as c:
        body = {"model": "m", "messages": [{"role": "user", "content": "prefix"}], "max_tokens": 5}
        await c.post("/v1/chat/completions", json=body, headers={"x-atfm-session": "s1", "x-atfm-class": "background", "x-atfm-tenant": "t"})
        r = await c.get("/session/s1/prompt")
        assert r.status_code == 200 and r.json() == {"session_id": "s1", "model": "m", "messages": body["messages"]}
        assert (await c.get("/session/nope/prompt")).status_code == 404
