"""mini-SWE-agent environment subclasses that route every tool through the sidecar.

The result the agent sees is the tool's own output; the sidecar only adds the state path
(progress/data events) and the optional launch gate for deferrable classes.
"""
from __future__ import annotations

import os
from .adapters import SidecarExecutor
from .config import DEFAULT_BACKENDS, SidecarConfig


class SidecarMixin:
    """Provides sidecar_execute for environment subclasses that carry a `sidecar` config."""

    sidecar: SidecarConfig

    def __init__(self, *, sidecar: SidecarConfig, **kwargs):
        super().__init__(**kwargs)
        self.sidecar = sidecar

    def sidecar_execute(self, command: str, cwd: str, timeout: float | None, argv_builder, host_cwd: str | None = "",
                        env: dict | None = None) -> dict:
        """`cwd` is the tool's working directory as the harness understands it (inside the container for
        Docker); `host_cwd` is where the launching subprocess runs on this host (default: same as cwd)."""
        output = SidecarExecutor(self.sidecar, lambda value: argv_builder(value)).execute(
            command, cwd=cwd if host_cwd == "" else host_cwd, timeout=timeout, env=env,
        )
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
        def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None) -> dict:
            command = action.get("command", "")
            cwd = cwd or self.config.cwd or os.getcwd()
            return self.sidecar_execute(command, cwd, timeout or self.config.timeout, lambda c: ["bash", "-lc", c],
                                        env=os.environ | dict(self.config.env or {}))

    class SidecarDockerEnvironment(SidecarMixin, DockerEnvironment):
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
