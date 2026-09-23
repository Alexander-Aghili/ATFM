from __future__ import annotations

import os
import queue as _queue
import signal
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass

from atfm.schema.events import ToolData, ToolEnd, ToolProgress, ToolStart

from .parsers import default_parsers


@dataclass
class ToolContext:
    session_id: str
    turn_index: int
    tool_name: str
    backend_id: str = "local"
    args_hash: str | None = None


@dataclass
class ToolResult:
    call_id: str
    output: bytes
    returncode: int
    duration_s: float
    timed_out: bool = False


_BUILD = ("make", "cmake", "ninja", "cargo build", "npm run build", "gradle")
_INSTALL = ("pip install", "npm install", "apt-get", "uv sync", "uv pip")


def classify_tool(command: str) -> str:
    c = command.strip()
    if "pytest" in c:
        return "pytest"
    if any(b in c for b in _BUILD):
        return "build"
    if any(i in c for i in _INSTALL):
        return "install"
    if "git clone" in c:
        return "clone"
    last = c.split("&&")[-1].strip()
    return last.split()[0] if last else "sh"


def _safe_publish(bus, e) -> None:
    try:
        bus.publish(e)
    except Exception:
        pass


def run_tool(cmd, ctx: ToolContext, bus, *, cwd=None, env=None, timeout: float | None = None, shell=None,
             parsers=None, clock=time.time) -> ToolResult:
    """Run a tool subprocess. The returned bytes are exactly what the tool wrote (stdout+stderr merged);
    the same stream feeds the parser chain, which emits progress and data events. Never raises on
    parser errors; kills the whole process group on timeout."""
    if shell is None:
        shell = isinstance(cmd, str)
    parsers = default_parsers() if parsers is None else parsers
    call_id = uuid.uuid4().hex[:16]
    t_start = clock()
    _safe_publish(bus, ToolStart(t=t_start, session_id=ctx.session_id, turn_index=ctx.turn_index, call_id=call_id,
                                 tool_name=ctx.tool_name, backend_id=ctx.backend_id, args_hash=ctx.args_hash))
    try:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
    except OSError:
        # The tool never launched: still close the state path so the board sees the phase end.
        _safe_publish(bus, ToolEnd(t=clock(), session_id=ctx.session_id, call_id=call_id, exit_status=-1, output_chars=0))
        raise
    assert proc.stdout is not None
    q: _queue.Queue = _queue.Queue()

    def _reader():
        try:
            for line in iter(proc.stdout.readline, b""):
                q.put(line)
        finally:
            q.put(None)

    threading.Thread(target=_reader, daemon=True).start()
    chunks: list[bytes] = []
    last_completed: float | None = None
    timed_out = False
    deadline = None if timeout is None else t_start + timeout
    try:
        while True:
            remaining = None if deadline is None else max(0.0, deadline - clock())
            try:
                line = q.get(timeout=0.5 if remaining is None else min(remaining, 0.5))
            except _queue.Empty:
                if deadline is not None and clock() > deadline:
                    timed_out = True
                    break
                continue
            if line is None:
                break
            chunks.append(line)
            now = clock()
            text = line.decode("utf-8", errors="replace").rstrip("\n")
            for p in parsers:
                try:
                    prog = p.feed(text, now)
                    if prog is not None:
                        if prog.get("completed") != last_completed:
                            last_completed = prog["completed"]
                            total = prog.get("total")
                            _safe_publish(bus, ToolProgress(t=now, session_id=ctx.session_id, call_id=call_id,
                                                            completed=float(prog["completed"]),
                                                            total=None if total is None else float(total),
                                                            phase=prog.get("phase")))
                        break
                    fd = getattr(p, "feed_data", None)
                    if fd is not None:
                        d = fd(text, now)
                        if d is not None:
                            _safe_publish(bus, ToolData(t=now, session_id=ctx.session_id, call_id=call_id,
                                                        metric=d["metric"], value=float(d["value"])))
                except Exception:
                    continue
    finally:
        if timed_out:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                proc.kill()
            # drain whatever the reader thread still delivers
            while True:
                try:
                    rest = q.get(timeout=1.0)
                except _queue.Empty:
                    break
                if rest is None:
                    break
                chunks.append(rest)
        proc.wait()
    output = b"".join(chunks)
    rc = -1 if timed_out else int(proc.returncode)
    t_end = clock()
    _safe_publish(bus, ToolEnd(t=t_end, session_id=ctx.session_id, call_id=call_id, exit_status=rc, output_chars=len(output)))
    return ToolResult(call_id=call_id, output=output, returncode=rc, duration_s=t_end - t_start, timed_out=timed_out)
