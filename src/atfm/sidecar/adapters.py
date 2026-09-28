"""Harness adapters beyond mini-SWE-agent (spec 7).

Two shapes. `SidecarExecutor` runs the tool subprocess itself through `run_tool`: byte-exact result path,
live progress, launch gate. `wrap_executor` wraps an executor the sidecar cannot stream (OpenHands'
terminal tool, Harbor's `environment.exec`): it emits tool.start, parses the returned output line by line
after the fact (progress timestamps are all at completion, a weaker signal), and emits tool.end, returning
the executor's result unchanged."""
from __future__ import annotations

import uuid
from typing import Callable

from atfm.schema.events import ToolEnd, ToolStart
from atfm.sidecar.core import ToolContext, _ToolEvents, _safe_publish, classify_tool, command_signature, run_tool
from atfm.sidecar.config import SidecarConfig
from atfm.sidecar.parsers import default_parsers


class SidecarExecutor:
    """Run commands through the sidecar's own subprocess wrapper. Result dict matches mini-SWE-agent's."""

    def __init__(self, cfg: SidecarConfig, argv_builder: Callable[[str], list[str]] | None = None):
        self.cfg = cfg
        self.argv_builder = argv_builder or (lambda c: ["bash", "-lc", c])

    def execute(self, command: str, cwd: str | None = None, timeout: float | None = None, env: dict | None = None) -> dict:
        cfg = self.cfg
        cfg.wait_for_gate()
        tool = classify_tool(command)
        ctx = ToolContext(session_id=cfg.session_id, turn_index=cfg.turn_index, tool_name=tool,
                          backend_id=cfg.backend_for(tool), args_hash=command_signature(command))
        cfg.turn_index += 1
        try:
            res = run_tool(self.argv_builder(command), ctx, cfg.bus, cwd=cwd or None, env=env, timeout=timeout,
                           shell=False, clock=cfg.clock)
        except OSError as e:
            return {"output": "", "returncode": -1, "exception_info": f"An error occurred while executing the command: {e}"}
        return {"output": res.output.decode("utf-8", errors="replace"), "returncode": res.returncode,
                "exception_info": "" if not res.timed_out else f"timeout after {timeout}s"}


def wrap_executor(fn: Callable, cfg: SidecarConfig, *, output_key="output", rc_key="returncode",
                  tuple_result: bool = False, extract: Callable | None = None) -> Callable:
    """Wrap a synchronous executor while preserving its result and exception semantics."""
    return _ExecutorWrapper(fn, cfg, output_key, rc_key, tuple_result, extract)


class _ExecutorWrapper:
    def __init__(self, fn, cfg, output_key, rc_key, tuple_result, extract):
        self.fn, self.cfg = fn, cfg
        self.output_key, self.rc_key = output_key, rc_key
        self.tuple_result, self.extract = tuple_result, extract

    def __call__(self, command: str, *args, **kwargs):
        call_id = self._start(command)
        try:
            result = self.fn(command, *args, **kwargs)
        except BaseException:
            self._end(call_id, self.cfg.clock(), -1, 0)
            raise
        self._complete(call_id, result, self.cfg.clock())
        return result

    def _start(self, command):
        cfg = self.cfg
        cfg.wait_for_gate()
        tool, call_id = classify_tool(command), uuid.uuid4().hex[:16]
        _safe_publish(cfg.bus, ToolStart(t=cfg.clock(), session_id=cfg.session_id, turn_index=cfg.turn_index,
                     call_id=call_id, tool_name=tool, backend_id=cfg.backend_for(tool), args_hash=command_signature(command)))
        cfg.turn_index += 1
        return call_id

    def _result_fields(self, result):
        if self.extract is not None:
            output, rc = self.extract(result)
        elif self.tuple_result:
            output, rc = result[0], result[1]
        else:
            output, rc = result.get(self.output_key, ''), result.get(self.rc_key, 0)
        text = output if isinstance(output, str) else bytes(output).decode('utf-8', errors='replace')
        return text, rc

    def _complete(self, call_id, result, now):
        try:
            text, rc = self._result_fields(result)
        except Exception:
            self._end(call_id, now, 0, 0)
            return
        ctx = ToolContext(self.cfg.session_id, self.cfg.turn_index, '')
        events = _ToolEvents(ctx, self.cfg.bus, call_id, default_parsers())
        for line in text.splitlines():
            events.feed(line, now)
        try:
            exit_status = int(rc) if rc is not None else 0
        except (TypeError, ValueError):
            exit_status = 0
        self._end(call_id, now, exit_status, len(text))

    def _end(self, call_id, now, status, chars):
        _safe_publish(self.cfg.bus, ToolEnd(t=now, session_id=self.cfg.session_id, call_id=call_id,
                                           exit_status=status, output_chars=chars))
