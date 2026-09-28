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
from atfm.sidecar.core import ToolContext, _safe_publish, classify_tool, command_signature, run_tool
from atfm.sidecar.config import SidecarConfig
from atfm.sidecar.events import publish_progress
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
    """Wrap `fn(command, **kw) -> result`. With `tuple_result`, the result is `(output, returncode)`; `extract`
    maps any other result shape to `(output, returncode)`. The result is returned unchanged whatever happens
    after the executor returns (spec 10: the result path is a pass-through)."""

    def wrapped(command: str, *args, **kwargs):
        cfg.wait_for_gate()
        tool = classify_tool(command)
        call_id = uuid.uuid4().hex[:16]
        t0 = cfg.clock()
        _safe_publish(cfg.bus, ToolStart(t=t0, session_id=cfg.session_id, turn_index=cfg.turn_index, call_id=call_id,
                                         tool_name=tool, backend_id=cfg.backend_for(tool), args_hash=command_signature(command)))
        cfg.turn_index += 1
        try:
            result = fn(command, *args, **kwargs)
        except BaseException:
            _safe_publish(cfg.bus, ToolEnd(t=cfg.clock(), session_id=cfg.session_id, call_id=call_id, exit_status=-1, output_chars=0))
            raise
        t1 = cfg.clock()
        try:
            if extract is not None:
                output, rc = extract(result)
            elif tuple_result:
                output, rc = result[0], result[1]
            else:
                output, rc = result.get(output_key, ""), result.get(rc_key, 0)
            text = output if isinstance(output, str) else bytes(output).decode("utf-8", errors="replace")
        except Exception:                          # unknown result shape: close the state path, hand the result back
            _safe_publish(cfg.bus, ToolEnd(t=t1, session_id=cfg.session_id, call_id=call_id, exit_status=0, output_chars=0))
            return result
        parsers, last_completed = default_parsers(), None
        for line in text.splitlines():
            last_completed = publish_progress(line, t1, parsers, cfg.bus, cfg.session_id, call_id, last_completed)
        try:
            exit_status = int(rc) if rc is not None else 0
        except (TypeError, ValueError):
            exit_status = 0
        _safe_publish(cfg.bus, ToolEnd(t=t1, session_id=cfg.session_id, call_id=call_id, exit_status=exit_status, output_chars=len(text)))
        return result

    return wrapped
