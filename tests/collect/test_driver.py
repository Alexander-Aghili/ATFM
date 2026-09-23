import time
from atfm.bus import InMemoryBus
from atfm.collect.jobs import CollectionSpec, JobSpec, TurnSpec
from atfm.collect.driver import run_collection, no_llm
from atfm.sidecar.minisweagent import SidecarConfig, SidecarMixin
from atfm.traces.sidecar import events_to_trace_table

class _LocalEnv(SidecarMixin):
    def __init__(self, cfg):
        self.sidecar = cfg
    def execute(self, action, cwd="", *, timeout=None):
        return self.sidecar_execute(action["command"], "/tmp", timeout, lambda c: ["bash", "-lc", c])
    def _check_finished(self, output):
        pass

def test_collection_records_sessions_and_tools(tmp_path):
    bus = InMemoryBus()
    spec = CollectionSpec(proxy_url=None, model="m", events_path=str(tmp_path / "ev.jsonl"), jobs=[
        JobSpec(name="fake-pytest", cls="background", tenant="t1", image=None, repeat=2,
                setup=["echo setup"],
                turns=[TurnSpec(cmd="printf 'collected 2 items\\na::t PASSED [ 50%%]\\nb::t PASSED [100%%]\\n'"),
                       TurnSpec(cmd="echo done")])])
    out = run_collection(spec, env_factory=lambda job, cfg: _LocalEnv(cfg), llm=no_llm(bus), bus=bus)
    assert out["sessions"] == 2 and out["tools"] == 6 and out["errors"] == 0
    t = events_to_trace_table(bus.drain())
    df = t.df
    assert df.session_id.nunique() == 2 and (df["class"] == "background").all()
    with_tools = df[df.tool_name.notna()]
    assert (with_tools["progress_events"].apply(len) >= 2).sum() == 2      # one pytest-shaped tool per session
    assert set(with_tools.tool_name) >= {"printf", "echo"}
