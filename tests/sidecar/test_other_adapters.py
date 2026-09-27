"""OpenHands and Harbor adapters (spec 7): the sidecar either runs the subprocess itself (byte-exact result
path, live progress) or wraps an executor it cannot stream, parsing its output after the fact."""
from atfm.bus import InMemoryBus
from atfm.sidecar.adapters import SidecarExecutor, wrap_executor
from atfm.sidecar.harbor import sidecar_exec
from atfm.sidecar.minisweagent import SidecarConfig
from atfm.sidecar.openhands import sidecar_terminal


def test_sidecar_executor_runs_the_subprocess_and_streams_progress():
    bus = InMemoryBus()
    ex = SidecarExecutor(SidecarConfig(session_id="oh1", cls="background", bus=bus))
    out = ex.execute("printf 'collected 2 items\\na::t PASSED [ 50%%]\\nb::t PASSED [100%%]\\n'", cwd="/tmp", timeout=10)
    assert out["returncode"] == 0 and out["output"].startswith("collected 2 items") and out["exception_info"] == ""
    kinds = [e.kind for e in bus.drain()]
    assert kinds[0] == "tool.start" and "tool.progress" in kinds and kinds[-1] == "tool.end"


def test_wrap_executor_keeps_output_byte_exact_and_emits_events_post_hoc():
    bus = InMemoryBus()
    calls = []

    def fake_exec(command: str, timeout: float | None = None) -> dict:
        calls.append(command)
        return {"output": "collected 3 items\nt::x PASSED [ 33%]\nt::y PASSED [ 66%]\nt::z FAILED [100%]\n=== 1 failed ===\n", "returncode": 1}

    wrapped = wrap_executor(fake_exec, SidecarConfig(session_id="oh2", cls="background", bus=bus), output_key="output", rc_key="returncode")
    res = wrapped("pytest -q", timeout=5)
    assert res == fake_exec("pytest -q") and calls == ["pytest -q", "pytest -q"]
    ev = bus.drain()
    assert ev[0].kind == "tool.start" and ev[0].tool_name == "pytest" and ev[-1].kind == "tool.end" and ev[-1].exit_status == 1
    prog = [e for e in ev if e.kind == "tool.progress"]
    assert prog and prog[-1].completed == 3 and prog[-1].total == 3 and all(e.t >= ev[0].t for e in prog)


def test_openhands_and_harbor_wrappers_match_their_harness_signatures():
    bus = InMemoryBus()
    cfg = SidecarConfig(session_id="h1", cls="interactive", bus=bus)
    oh = sidecar_terminal(lambda command, timeout=None: {"output": "ok\n", "returncode": 0, "extra": 1}, cfg)
    assert oh("ls", timeout=1) == {"output": "ok\n", "returncode": 0, "extra": 1}
    hb = sidecar_exec(lambda cmd, timeout=None: ("stage 1/2\nstage 2/2\n", 0), cfg)
    assert hb("run", timeout=1) == ("stage 1/2\nstage 2/2\n", 0)
    kinds = [e.kind for e in bus.drain()]
    assert kinds.count("tool.start") == 2 and kinds.count("tool.end") == 2 and "tool.progress" in kinds
