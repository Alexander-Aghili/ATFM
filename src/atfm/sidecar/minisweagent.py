"""mini-SWE-agent environment subclasses that route every tool through the sidecar.

The result the agent sees is the tool's own output; the sidecar only adds the state path
(progress/data events) and the optional launch gate for deferrable classes.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Callable

from .core import ToolContext, classify_tool, run_tool
from .gate import gate_allowed_at

DEFAULT_BACKENDS = {"pytest": "ci", "build": "ci", "install": "pkg", "clone": "git"}


@dataclass
class SidecarConfig:
    session_id: str
    tenant: str = "t0"
    cls: str = "background"
    bus: object = None
    gate_url: str | None = None
    deferrable: bool | None = None
    backend_map: dict[str, str] | None = None
    max_gate_wait_s: float = 600.0
    clock: Callable[[], float] = time.time
    turn_index: int = field(default=0)

    def __post_init__(self):
        if self.deferrable is None:
            self.deferrable = self.cls == "background"
        if self.backend_map is None:
            self.backend_map = dict(DEFAULT_BACKENDS)
        if self.bus is None:
            from atfm.bus import InMemoryBus
            self.bus = InMemoryBus()

    def backend_for(self, tool_name: str) -> str:
        return self.backend_map.get(tool_name, "local")


class SidecarMixin:
    """Provides sidecar_execute for environment subclasses that carry a `sidecar` config."""

    sidecar: SidecarConfig

    def sidecar_execute(self, command: str, cwd: str, timeout: float | None, argv_builder, host_cwd: str | None = "",
                        env: dict | None = None) -> dict:
        """`cwd` is the tool's working directory as the harness understands it (inside the container for
        Docker); `host_cwd` is where the launching subprocess runs on this host (default: same as cwd)."""
        cfg = self.sidecar
        if host_cwd == "":
            host_cwd = cwd
        tool = classify_tool(command)
        if cfg.deferrable:
            allowed = gate_allowed_at(cfg.gate_url, cfg.session_id, "tool")
            if allowed is not None:
                wait = min(max(0.0, allowed - cfg.clock()), cfg.max_gate_wait_s)
                if wait > 0:
                    time.sleep(wait)
        ctx = ToolContext(session_id=cfg.session_id, turn_index=cfg.turn_index, tool_name=tool,
                          backend_id=cfg.backend_for(tool))
        cfg.turn_index += 1
        try:
            res = run_tool(argv_builder(command), ctx, cfg.bus, cwd=host_cwd or None, env=env, timeout=timeout,
                           shell=False, clock=cfg.clock)
        except OSError as e:
            output = {"output": "", "returncode": -1,
                      "exception_info": f"An error occurred while executing the command: {e}"}
            self._check_finished(output)
            return output
        output = {"output": res.output.decode("utf-8", errors="replace"), "returncode": res.returncode,
                  "exception_info": "" if not res.timed_out else f"timeout after {timeout}s"}
        self._check_finished(output)
        return output


try:
    from minisweagent.environments.docker import DockerEnvironment
    from minisweagent.environments.local import LocalEnvironment
except ImportError as _e:  # pragma: no cover
    LocalEnvironment = DockerEnvironment = None
    _import_error = _e
else:
    _import_error = None

if LocalEnvironment is not None:

    class SidecarLocalEnvironment(SidecarMixin, LocalEnvironment):
        def __init__(self, *, sidecar: SidecarConfig, **kwargs):
            super().__init__(**kwargs)
            self.sidecar = sidecar

        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
            command = action.get("command", "")
            cwd = cwd or self.config.cwd or os.getcwd()
            return self.sidecar_execute(command, cwd, timeout or self.config.timeout, lambda c: ["bash", "-lc", c],
                                        env=os.environ | dict(self.config.env or {}))

    class SidecarDockerEnvironment(SidecarMixin, DockerEnvironment):
        def __init__(self, *, sidecar: SidecarConfig, **kwargs):
            super().__init__(**kwargs)
            self.sidecar = sidecar

        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
            command = action.get("command", "")
            cwd = cwd or self.config.cwd
            assert self.container_id, "Container not started"

            def argv(c):
                cmd = [self.config.executable, "exec", "-w", cwd]
                for key in self.config.forward_env:
                    if (value := os.getenv(key)) is not None:
                        cmd.extend(["-e", f"{key}={value}"])
                for key, value in (self.config.env or {}).items():
                    cmd.extend(["-e", f"{key}={value}"])
                interpreter = list(getattr(self.config, "interpreter", None) or ["bash", "-lc"])
                return cmd + [self.container_id, *interpreter, c]

            return self.sidecar_execute(command, cwd, timeout or self.config.timeout, argv, host_cwd=None)

else:

    def __getattr__(name):  # pragma: no cover
        if name in ("SidecarLocalEnvironment", "SidecarDockerEnvironment"):
            raise ImportError(f"{name} needs mini-swe-agent: uv sync --extra harness ({_import_error})")
        raise AttributeError(name)
