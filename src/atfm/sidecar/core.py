from __future__ import annotations

import os
import queue as _queue
import re
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


_PYTEST_CMD = re.compile(r"(^|&&|;|\|)\s*(python(3)?( -m)?\s+)?pytest\b")


_NUM = re.compile(r"^[0-9]+([.][0-9]+)?$")


def command_signature(command: str) -> str:
    """Normalized command text for keying progress curves: standalone numbers (sizes, counts, rates)
    become `N`, while digits attached to a flag (`-j2`) or a redirection (`2>&1`) stay, because those
    are modes that change the shape of progress rather than its length. At most 64 characters."""
    toks = ["N" if _NUM.match(t) else t for t in command.split()]
    sig = " ".join(toks)
    if len(sig) > 64:
        import hashlib
        sig = sig[:52] + "~" + hashlib.sha1(sig.encode()).hexdigest()[:8]
    return sig


def classify_tool(command: str) -> str:
    """Coarse tool class from the command text. Installs and clones are checked first so that
    `pip install pytest` or `apt-get install cmake` are not mistaken for test or build runs."""
    c = command.strip()
    if any(i in c for i in _INSTALL):
        return "install"
    if "git clone" in c:
        return "clone"
    if _PYTEST_CMD.search(c):
        return "pytest"
    if any(b in c for b in _BUILD):
        return "build"
    last = c.split("&&")[-1].strip()
    return last.split()[0] if last else "sh"


def _safe_publish(bus, e) -> None:
    try:
        bus.publish(e)
    except Exception:
        pass


def run_tool(cmd, ctx: ToolContext, bus, *, cwd=None, env=None, timeout: float | None = None, shell=None,
             parsers=None, clock=time.time) -> ToolResult:
    """Capture merged output unchanged; parser errors are isolated and timeouts kill the process group."""
    run = _ToolRun(ctx, bus, default_parsers() if parsers is None else parsers, clock)
    proc = run.launch(cmd, cwd, env, isinstance(cmd, str) if shell is None else shell)
    return run.collect(proc, timeout)


def _read_lines(stream, output):
    try:
        for line in iter(stream.readline, b''):
            output.put(line)
    finally:
        output.put(None)


def _kill_and_drain(proc, output, chunks):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()
    while True:
        try:
            rest = output.get(timeout=1.0)
        except _queue.Empty:
            break
        if rest is None:
            break
        chunks.append(rest)


class _ToolRun:
    def __init__(self, ctx, bus, parsers, clock):
        self.ctx, self.bus, self.parsers, self.clock = ctx, bus, parsers, clock
        self.call_id, self.started = uuid.uuid4().hex[:16], clock()
        self.events = _ToolEvents(ctx, bus, self.call_id, parsers)
        self.chunks = []

    def launch(self, cmd, cwd, env, shell):
        ctx = self.ctx
        _safe_publish(self.bus, ToolStart(t=self.started, session_id=ctx.session_id, turn_index=ctx.turn_index,
                     call_id=self.call_id, tool_name=ctx.tool_name, backend_id=ctx.backend_id, args_hash=ctx.args_hash))
        try:
            return subprocess.Popen(cmd, cwd=cwd, env=env, shell=shell, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, start_new_session=True)
        except Exception:
            self._end(-1, b'')
            raise

    def collect(self, proc, timeout):
        assert proc.stdout is not None
        output = _queue.Queue()
        threading.Thread(target=_read_lines, args=(proc.stdout, output), daemon=True).start()
        deadline = None if timeout is None else self.started + timeout
        timed_out = False
        try:
            timed_out = self._consume(output, deadline)
        finally:
            if timed_out:
                _kill_and_drain(proc, output, self.chunks)
            proc.wait()
        data = b''.join(self.chunks)
        rc = -1 if timed_out else int(proc.returncode)
        duration = self._end(rc, data)
        return ToolResult(self.call_id, data, rc, duration, timed_out)

    def _consume(self, output, deadline):
        while True:
            remaining = None if deadline is None else max(0.0, deadline - self.clock())
            try:
                line = output.get(timeout=0.5 if remaining is None else min(remaining, 0.5))
            except _queue.Empty:
                if deadline is not None and self.clock() > deadline:
                    return True
                continue
            if line is None:
                return False
            self.chunks.append(line)
            self.events.feed(line.decode('utf-8', errors='replace').rstrip('\n'), self.clock())

    def _end(self, rc, output):
        ended = self.clock()
        _safe_publish(self.bus, ToolEnd(t=ended, session_id=self.ctx.session_id, call_id=self.call_id,
                                       exit_status=rc, output_chars=len(output)))
        return ended - self.started


class _ToolEvents:
    """Share parser ordering, duplicate suppression, and failure isolation across adapters."""

    def __init__(self, ctx, bus, call_id, parsers):
        self.ctx, self.bus, self.call_id, self.parsers = ctx, bus, call_id, parsers
        self.last_completed = None

    def feed(self, text, now):
        for parser in self.parsers:
            try:
                progress = parser.feed(text, now)
                if progress is not None:
                    self._progress(progress, now)
                    break
                self._data(parser, text, now)
            except Exception:
                continue

    def _progress(self, progress, now):
        if progress.get('completed') == self.last_completed:
            return
        self.last_completed = progress['completed']
        total = progress.get('total')
        _safe_publish(self.bus, ToolProgress(t=now, session_id=self.ctx.session_id, call_id=self.call_id,
                     completed=float(progress['completed']), total=None if total is None else float(total),
                     phase=progress.get('phase')))

    def _data(self, parser, text, now):
        feed = getattr(parser, 'feed_data', None)
        if feed is None:
            return
        data = feed(text, now)
        if data is not None:
            _safe_publish(self.bus, ToolData(t=now, session_id=self.ctx.session_id, call_id=self.call_id,
                         metric=data['metric'], value=float(data['value'])))

