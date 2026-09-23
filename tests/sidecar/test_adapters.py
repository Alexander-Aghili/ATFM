import json, threading, time
from http.server import BaseHTTPRequestHandler, HTTPServer
import pytest
from atfm.bus import InMemoryBus
from atfm.sidecar.gate import gate_allowed_at
from atfm.sidecar.minisweagent import SidecarConfig, SidecarMixin

class _Gate(BaseHTTPRequestHandler):
    allowed_at = None
    def do_POST(self):
        n = int(self.headers.get("content-length", 0)); body = json.loads(self.rfile.read(n))
        self.send_response(200); self.send_header("content-type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps({"allowed_at": _Gate.allowed_at, "echo": body}).encode())
    def log_message(self, *a): pass

@pytest.fixture
def gate_server():
    srv = HTTPServer(("127.0.0.1", 0), _Gate)
    th = threading.Thread(target=srv.serve_forever, daemon=True); th.start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()

def test_gate_fail_open_and_value(gate_server):
    assert gate_allowed_at(None, "s", "tool") is None
    assert gate_allowed_at("http://127.0.0.1:1", "s", "tool", timeout_s=0.2) is None
    _Gate.allowed_at = 123.0
    assert gate_allowed_at(gate_server, "s", "tool") == 123.0

class _Env(SidecarMixin):
    def __init__(self, cfg):
        self.sidecar = cfg
    def _check_finished(self, output):
        pass

def test_mixin_runs_tool_and_publishes(gate_server):
    bus = InMemoryBus()
    _Gate.allowed_at = time.time() + 0.3
    cfg = SidecarConfig(session_id="s9", cls="background", bus=bus, gate_url=gate_server)
    env = _Env(cfg)
    t0 = time.time()
    out = env.sidecar_execute("printf 'collected 1 items\\nx::t PASSED [100%%]\\n'", cwd="/tmp", timeout=10,
                              argv_builder=lambda c: ["bash", "-lc", c])
    assert out["returncode"] == 0 and out["output"].startswith("collected 1 items") and out["exception_info"] == ""
    assert time.time() - t0 >= 0.25
    ev = bus.drain()
    assert ev[0].kind == "tool.start" and ev[0].tool_name == "printf"
    assert any(e.kind == "tool.progress" for e in ev) and ev[-1].kind == "tool.end"
    assert env.sidecar.turn_index == 1

def test_interactive_skips_gate():
    bus = InMemoryBus()
    cfg = SidecarConfig(session_id="i1", cls="interactive", bus=bus, gate_url="http://127.0.0.1:1")
    t0 = time.time()
    out = _Env(cfg).sidecar_execute("echo hi", cwd="/tmp", timeout=5, argv_builder=lambda c: ["bash", "-lc", c])
    assert out["output"] == "hi\n" and time.time() - t0 < 1.0

def test_minisweagent_classes_importable_or_skipped():
    pytest.importorskip("minisweagent")
    from atfm.sidecar.minisweagent import SidecarLocalEnvironment
    env = SidecarLocalEnvironment(sidecar=SidecarConfig(session_id="l1", bus=InMemoryBus()))
    out = env.execute({"command": "echo local"}, cwd="/tmp")
    assert out["output"] == "local\n" and out["returncode"] == 0

def test_mixin_host_cwd_separate_from_container_cwd_and_failure_is_a_result():
    bus = InMemoryBus()
    env = _Env(SidecarConfig(session_id="d1", cls="interactive", bus=bus))
    # container cwd "/w" does not exist on the host; with host_cwd=None the tool still runs
    out = env.sidecar_execute("echo ok", cwd="/w", timeout=5, argv_builder=lambda c: ["bash", "-lc", c], host_cwd=None)
    assert out["returncode"] == 0 and out["output"] == "ok\n"
    # a launch failure becomes a -1 result with exception_info, never an exception
    out = env.sidecar_execute("echo ok", cwd="/nonexistent-atfm-dir", timeout=5, argv_builder=lambda c: ["bash", "-lc", c])
    assert out["returncode"] == -1 and "No such file" in out["exception_info"]
    assert [e.kind for e in bus.drain()][-1] == "tool.end"

def test_local_environment_forwards_config_env():
    pytest.importorskip("minisweagent")
    from atfm.sidecar.minisweagent import SidecarLocalEnvironment
    env = SidecarLocalEnvironment(sidecar=SidecarConfig(session_id="e1", bus=InMemoryBus()), env={"ATFM_PROBE": "42"})
    assert env.execute({"command": "echo $ATFM_PROBE"}, cwd="/tmp")["output"] == "42\n"
