"""LMCache actuator: our placement decisions executed by LMCache's controller (pin / move / lookup) instead
of keep-alive touches. Client and tokenizer are injected; nothing here talks to a real LMCache."""
import pytest

from atfm.control import TierDirective, TouchDirective
from atfm.control.lmcache import LMCacheActuator, LMCacheConfig, PromptTokens


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}
    def json(self):
        return self._body


class FakeClient:
    def __init__(self, fail=False):
        self.calls, self.fail = [], fail
        self._n = 0
    def post(self, url, json=None, **kw):
        self.calls.append((url, json))
        if self.fail:
            raise ConnectionError("lmcache down")
        self._n += 1
        if url.endswith("/lookup"):          # documented response: instance-keyed (location, matched prefix length)
            return FakeResponse(200, {"event_id": "ev", "vllm-0": ["LocalCPUBackend", len(json["tokens"]) // 2]})
        if url.endswith("/check_finish"):
            return FakeResponse(200, {"status": "finished"})
        return FakeResponse(200, {"event_id": f"ev{self._n}", "num_tokens": len(json.get("tokens", []))})


def _tokens():
    return PromptTokens(tokenize=lambda messages: [len(m["content"]) for m in messages] * 3,
                        prompt_source=lambda sid: {"sess": [{"role": "user", "content": "hello"}, {"role": "assistant", "content": "hi!"}]}.get(sid))


def test_touch_directive_becomes_a_gpu_pin_with_the_session_prefix_tokens():
    client = FakeClient()
    act = LMCacheActuator(LMCacheConfig(url="http://lmcache:9000", instance_id="vllm-0", gpu_location="LocalGPUBackend",
                                        cpu_location="LocalCPUBackend"), client=client, tokens=_tokens())
    out = act.apply_touch(TouchDirective(session_id="sess", worker_id="w0", eta_q50=4.0, expires_at=100.0), now=50.0)
    assert out == {"ok": True, "event_id": "ev1", "num_tokens": 6}
    url, body = client.calls[-1]
    assert url == "http://lmcache:9000/pin" and body == {"instance_id": "vllm-0", "location": "LocalGPUBackend", "tokens": [5, 3, 5, 3, 5, 3]}
    assert act.pinned["sess"] == (100.0, "LocalGPUBackend")                      # remembered with its expiry


def test_tier_directives_map_to_pin_move_and_unpin():
    client = FakeClient()
    act = LMCacheActuator(LMCacheConfig(url="http://lmcache:9000", instance_id="vllm-0"), client=client, tokens=_tokens())
    act.apply_tier(TierDirective(session_id="sess", action="pin", tier="gpu", eta_q10=1.0, eta_q90=5.0, expires_at=10.0), now=0.0)
    act.apply_tier(TierDirective(session_id="sess", action="demote", tier="cpu", eta_q10=30.0, eta_q90=90.0, expires_at=60.0), now=1.0)
    act.apply_tier(TierDirective(session_id="sess", action="promote", tier="gpu", eta_q10=2.0, eta_q90=8.0, expires_at=70.0), now=2.0)
    urls = [u.rsplit("/", 1)[1] for u, _ in client.calls]
    assert urls == ["pin", "move", "move"]
    assert client.calls[1][1] == {"src": {"instance_id": "vllm-0", "location": "LocalGPUBackend"},
                                  "dst": {"instance_id": "vllm-0", "location": "LocalCPUBackend"}, "tokens": [5, 3, 5, 3, 5, 3]}
    act.apply_tier(TierDirective(session_id="sess", action="demote", tier="disk", eta_q10=500.0, eta_q90=900.0, expires_at=80.0), now=3.0)
    assert client.calls[-1][1]["dst"]["location"] == "LocalDiskBackend"        # tier names map through the config


def test_expired_pins_are_released_and_failures_are_fail_open():
    client = FakeClient()
    act = LMCacheActuator(LMCacheConfig(url="http://lmcache:9000", instance_id="vllm-0"), client=client, tokens=_tokens())
    act.apply_touch(TouchDirective(session_id="sess", expires_at=100.0), now=50.0)
    assert act.release_expired(now=99.0) == [] and act.release_expired(now=101.0) == ["sess"]
    assert client.calls[-1][0].endswith("/unpin") and client.calls[-1][1]["tokens"] and "sess" not in act.pinned
    dead = LMCacheActuator(LMCacheConfig(url="http://lmcache:9000", instance_id="vllm-0"), client=FakeClient(fail=True), tokens=_tokens())
    assert dead.apply_touch(TouchDirective(session_id="sess", expires_at=100.0), now=0.0)["ok"] is False and dead.errors == 1
    unknown = LMCacheActuator(LMCacheConfig(url="http://lmcache:9000", instance_id="vllm-0"), client=client, tokens=_tokens())
    assert unknown.apply_touch(TouchDirective(session_id="never", expires_at=100.0), now=0.0) == {"ok": False, "reason": "no prompt"}


def test_lookup_reports_where_a_session_prefix_lives():
    act = LMCacheActuator(LMCacheConfig(url="http://lmcache:9000", instance_id="vllm-0"), client=FakeClient(), tokens=_tokens())
    assert act.lookup("sess") == [{"instance_id": "vllm-0", "location": "LocalCPUBackend", "hit_tokens": 3}]
    assert act.lookup("never") == []


def test_prompt_tokens_caches_per_session_and_refreshes_on_new_prompt():
    calls = []
    src = {"s": [{"role": "user", "content": "a"}]}
    pt = PromptTokens(tokenize=lambda m: (calls.append(1), [1] * len(m))[1], prompt_source=lambda sid: src.get(sid))
    assert pt.get("s") == [1] and pt.get("s") == [1] and len(calls) == 1
    src["s"] = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
    assert pt.get("s") == [1, 1] and len(calls) == 2
