import os, sys, time
from atfm.bus import InMemoryBus
from atfm.sidecar.core import run_tool, ToolContext

CTX = ToolContext(session_id="s1", turn_index=0, tool_name="pytest", backend_id="local")

def test_output_is_byte_exact_and_events_emitted():
    bus = InMemoryBus()
    script = "import sys; sys.stdout.buffer.write(b'collected 2 items\\n'); sys.stdout.buffer.write(b'\\xff\\xfe raw bytes\\n'); sys.stdout.buffer.write(b'a::t PASSED [ 50%]\\n'); sys.stdout.flush(); sys.stderr.write('warn\\n'); sys.stderr.flush(); sys.stdout.buffer.write(b'b::t PASSED [100%]\\n')"
    res = run_tool([sys.executable, "-u", "-c", script], CTX, bus, shell=False)
    assert res.returncode == 0 and res.timed_out is False
    assert b"\xff\xfe raw bytes\n" in res.output and b"warn\n" in res.output and res.output.startswith(b"collected 2 items\n")
    kinds = [e.kind for e in bus.drain()]
    assert kinds[0] == "tool.start" and kinds[-1] == "tool.end" and kinds.count("tool.progress") == 3

def test_progress_is_deduplicated():
    bus = InMemoryBus()
    res = run_tool("printf '[ 10%%] a\\n[ 10%%] b\\n[ 20%%] c\\n'", CTX, bus)
    assert res.returncode == 0
    prog = [e for e in bus.drain() if e.kind == "tool.progress"]
    assert [e.completed for e in prog] == [10.0, 20.0]

def test_timeout_kills_and_reports():
    bus = InMemoryBus()
    t0 = time.time()
    res = run_tool([sys.executable, "-u", "-c", "import time,sys; print('start', flush=True); time.sleep(30)"], CTX, bus, shell=False, timeout=1.0)
    assert res.timed_out and res.returncode == -1 and b"start" in res.output and time.time() - t0 < 5
    end = [e for e in bus.drain() if e.kind == "tool.end"][0]
    assert end.exit_status == -1

def test_parser_exception_does_not_break_tool():
    class Bad:
        def feed(self, line, t):
            raise RuntimeError("boom")
    bus = InMemoryBus()
    res = run_tool("echo hello", CTX, bus, parsers=[Bad()])
    assert res.returncode == 0 and res.output == b"hello\n"

def test_launch_failure_emits_tool_end_and_raises():
    import pytest
    bus = InMemoryBus()
    with pytest.raises(OSError):
        run_tool(["bash", "-lc", "echo x"], CTX, bus, shell=False, cwd="/nonexistent-atfm-dir")
    kinds = [(e.kind, getattr(e, "exit_status", None)) for e in bus.drain()]
    assert kinds == [("tool.start", None), ("tool.end", -1)]

def test_timeout_kills_grandchildren(tmp_path):
    pidfile = tmp_path / "pid"
    bus = InMemoryBus()
    res = run_tool(f"sleep 30 & echo $! > {pidfile}; wait", CTX, bus, timeout=1.0)
    assert res.timed_out
    pid = int(pidfile.read_text().strip())
    time.sleep(0.2)
    assert not os.path.exists(f"/proc/{pid}"), "grandchild survived the process-group kill"

def test_launch_failure_non_oserror_still_emits_tool_end():
    import pytest
    bus = InMemoryBus()
    with pytest.raises(ValueError):
        run_tool(["bash", "-lc", "echo\x00x"], CTX, bus, shell=False)
    assert [e.kind for e in bus.drain()] == ["tool.start", "tool.end"]
