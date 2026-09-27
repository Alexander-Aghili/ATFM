"""The control loop routes touches and tier directives to LMCache when an actuator is attached."""
from atfm.control.lmcache import LMCacheActuator, LMCacheConfig, PromptTokens
from atfm.control.loop import ControlLoop
from tests.control.test_lmcache import FakeClient as LMClient
from tests.control.test_loop import DIRECTIVES, FakeClient, FakeResponse


def test_loop_pins_through_lmcache_instead_of_touching_the_proxy(tmp_path):
    http = FakeClient({("POST", "http://board/tick"): FakeResponse(200, {"sessions": 2}),
                       ("POST", "http://board/directives"): FakeResponse(200, DIRECTIVES),
                       ("POST", "http://proxy/directives"): FakeResponse(200, {"ok": True})})
    lm = LMClient()
    act = LMCacheActuator(LMCacheConfig(url="http://lmcache:9000", instance_id="vllm-0"), client=lm,
                          tokens=PromptTokens(tokenize=lambda m: [7] * len(m), prompt_source=lambda sid: [{"role": "user", "content": "x"}]))
    loop = ControlLoop(board_url="http://board", proxy_url="http://proxy", client=http, lmcache=act, log_path=tmp_path / "c.jsonl")
    s = loop.step(now=100.0)
    assert s["holds"] == 1 and s["touches"] == 0 and s["pins"] == 1 and s["tier_applied"] == 1 and s["errors"] == 0
    assert not any(u == "http://proxy/touch" for _, u, _ in http.calls)        # no touch when LMCache executes placement
    assert [u.rsplit("/", 1)[1] for u, _ in lm.calls] == ["pin", "move"]      # touch -> pin; tier demote -> move
