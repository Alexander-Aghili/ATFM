import os, threading, time
import pytest
pytestmark = pytest.mark.skipif(os.environ.get("ATFM_DYNAMO") != "1", reason="needs the dynamo extra and ATFM_DYNAMO=1")

def test_collection_through_proxy_and_mocker(tmp_path):
    import uvicorn, urllib.request, json
    from atfm.dynamo.local import LocalDynamo
    from atfm.proxy.app import create_app
    from atfm.proxy.config import ProxyConfig
    from atfm.bus import read_events
    from atfm.collect.driver import run_collection
    from atfm.collect.jobs import CollectionSpec, JobSpec, TurnSpec
    from atfm.traces.sidecar import events_to_trace_table
    from atfm.sidecar.minisweagent import SidecarLocalEnvironment
    events = tmp_path / "ev.jsonl"
    with LocalDynamo(port=8791, log_dir=str(tmp_path)) as d:
        app = create_app(ProxyConfig(upstream_url=d.base_url, window=2, events_path=str(events)))
        server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=8792, log_level="warning"))
        th = threading.Thread(target=server.run, daemon=True); th.start()
        time.sleep(1.0)
        peak, stop = _observe_peak()
        out = _collect(d, events)
        stop[0] = True; server.should_exit = True
    _assert_collection(out, events, peak)


def _assert_collection(out, events, peak):
    from atfm.bus import read_events
    from atfm.traces.sidecar import events_to_trace_table
    assert out["errors"] == 0 and out["sessions"] == 3
    ev = read_events(events)
    reqs = [e for e in ev if e.kind == "llm.request"]
    assert reqs and all(e.hints.get("strict_priority") in (0, 1, 2) for e in reqs)
    assert any(e.hints.get("strict_priority") == 2 for e in reqs)
    assert 1 <= peak[0] <= 2
    df = events_to_trace_table(ev).df
    tools = df[df.tool_name.notna()]
    assert (tools["t_tool_end"] > tools["t_tool_start"]).all()


def _collect(d, events):
    from atfm.collect.driver import run_collection
    from atfm.collect.jobs import CollectionSpec, JobSpec, TurnSpec
    from atfm.sidecar.minisweagent import SidecarLocalEnvironment
    spec = CollectionSpec(proxy_url="http://127.0.0.1:8792", model=d.model, events_path=str(events), concurrency=3, jobs=[
        JobSpec(name="a", cls="background", tenant="t", image=None, repeat=2, turns=[TurnSpec(cmd="sleep 0.2; echo a")] * 3),
        JobSpec(name="b", cls="interactive", tenant="t", image=None, repeat=1, deadline_s=5,
                turns=[TurnSpec(cmd="printf 'collected 1 items\\nx::t PASSED [100%%]\\n'")] * 3)])
    out = run_collection(spec, env_factory=lambda job, cfg: SidecarLocalEnvironment(sidecar=cfg, timeout=60, cwd="/tmp"))
    return out


def _observe_peak():
    import json, urllib.request
    peak = [0]; stop = [False]
    def poll():
        while not stop[0]:
            try:
                st = json.loads(urllib.request.urlopen("http://127.0.0.1:8792/state", timeout=1).read())
                peak[0] = max(peak[0], st["in_flight"])
            except Exception:
                pass
            time.sleep(0.02)
    pt = threading.Thread(target=poll, daemon=True); pt.start()
    return peak, stop
