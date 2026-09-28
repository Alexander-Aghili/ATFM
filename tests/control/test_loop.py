"""The control loop (deploy): pull directives from the board, push holds and touches to the proxy, log the
rest. Fail-open: a dead board or proxy never raises out of a step."""
import json

from atfm.control.loop import ControlLoop


class FakeResponse:
    def __init__(self, status=200, body=None):
        self.status_code, self._body = status, body or {}
    def json(self):
        return self._body


class FakeClient:
    """Minimal sync HTTP client: routes by (method, url)."""
    def __init__(self, routes):
        self.routes, self.calls = routes, []
    def get(self, url, **kw):
        self.calls.append(("GET", url, None))
        r = self.routes.get(("GET", url))
        if isinstance(r, Exception):
            raise r
        return r or FakeResponse(404)
    def post(self, url, json=None, **kw):
        self.calls.append(("POST", url, json))
        r = self.routes.get(("POST", url))
        if isinstance(r, Exception):
            raise r
        return r or FakeResponse(404)


DIRECTIVES = {"t": 100.0,
              "holds": [{"kind": "hold", "session_id": "bg1", "release_not_before": 130.0, "reason": "gdp", "expires_at": 130.0, "tenant": "t1"}],
              "touches": [{"kind": "touch", "session_id": "it1", "worker_id": "w0", "eta_q50": 4.0, "expires_at": 130.0}],
              "tier": [{"kind": "tier", "session_id": "bg2", "action": "demote", "tier": "cpu", "eta_q10": 20.0, "eta_q90": 90.0, "expires_at": 140.0}],
              "replica": {"kind": "replica", "replicas_at_least": 2, "horizon_s": 300.0, "expires_at": 400.0, "demand_q90_blocks": 1500.0}}


def test_step_pushes_holds_and_touches_and_logs_tier_and_replica(tmp_path):
    client = FakeClient({("POST", "http://board/tick"): FakeResponse(200, {"sessions": 3}),
                         ("POST", "http://board/directives"): FakeResponse(200, DIRECTIVES),
                         ("POST", "http://proxy/directives/batch"): FakeResponse(200, {"ok": True, "applied": 1, "expired": 0}),
                         ("POST", "http://proxy/touch"): FakeResponse(200, {"ok": True, "prompt_tokens": 400})})
    loop = ControlLoop(board_url="http://board", proxy_url="http://proxy", client=client, log_path=tmp_path / "control.jsonl")
    s = loop.step(now=100.0)
    assert s == {"sessions": 3, "holds": 1, "touches": 1, "touch_tokens": 400, "tier": 1, "replica": 2, "errors": 0, "pins": 0, "tier_applied": 0}
    posted = [(u, j) for m, u, j in client.calls if m == "POST" and u.startswith("http://proxy")]
    assert posted[0] == ("http://proxy/directives/batch", {"holds": [{"session_id": "bg1", "release_not_before": 130.0, "reason": "gdp", "expires_at": 130.0}]})
    assert posted[1] == ("http://proxy/touch", {"session_id": "it1"})
    rec = [json.loads(l) for l in (tmp_path / "control.jsonl").read_text().splitlines()]
    assert rec[-1]["tier"][0]["session_id"] == "bg2" and rec[-1]["replica"]["replicas_at_least"] == 2 and rec[-1]["t"] == 100.0


def test_expired_directives_are_not_pushed_and_failures_are_counted_not_raised(tmp_path):
    stale = dict(DIRECTIVES)
    stale["holds"] = [dict(DIRECTIVES["holds"][0], expires_at=90.0)]        # already expired at now=100
    client = FakeClient({("POST", "http://board/tick"): FakeResponse(200, {"sessions": 1}),
                         ("POST", "http://board/directives"): FakeResponse(200, stale),
                         ("POST", "http://proxy/directives/batch"): FakeResponse(200, {"ok": True, "applied": 1, "expired": 0}),
                         ("POST", "http://proxy/touch"): ConnectionError("proxy down")})
    loop = ControlLoop(board_url="http://board", proxy_url="http://proxy", client=client)
    s = loop.step(now=100.0)
    assert s["holds"] == 0 and s["touches"] == 0 and s["errors"] == 1
    dead = FakeClient({("POST", "http://board/tick"): ConnectionError("board down")})
    assert ControlLoop(board_url="http://board", proxy_url="http://proxy", client=dead).step(now=1.0)["errors"] == 1


def test_holds_are_chunked_and_non_success_responses_count_as_errors():
    directives = dict(DIRECTIVES, touches=[], tier=[], replica=None)
    directives['holds'] = [dict(DIRECTIVES['holds'][0], session_id=str(i)) for i in range(5)]
    client = FakeClient({('POST', 'b/tick'): FakeResponse(200, {'sessions': 5}),
                         ('POST', 'b/directives'): FakeResponse(200, directives)})
    original = client.post
    def post(url, json=None, **kwargs):
        if url == 'p/directives/batch':
            client.calls.append(('POST', url, json))
            if json['holds'][0]['session_id'] == '2':
                return FakeResponse(503)
            return FakeResponse(200, {'ok': True, 'applied': len(json['holds']), 'expired': 0})
        return original(url, json=json, **kwargs)
    client.post = post
    result = ControlLoop('b', 'p', client=client, hold_batch_size=2).step(now=100)
    assert result['holds'] == 3 and result['errors'] == 1
    batches = [payload['holds'] for _, url, payload in client.calls if url == 'p/directives/batch']
    assert list(map(len, batches)) == [2, 2, 1]
