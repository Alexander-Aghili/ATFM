"""OpenHands adapter (spec 7): wrap the terminal tool's `execute(command, timeout=...) -> {"output", "returncode", ...}`
so every command emits tool events. The result dict is returned untouched."""
from __future__ import annotations

from typing import Callable

from atfm.sidecar.adapters import SidecarExecutor, wrap_executor
from atfm.sidecar.config import SidecarConfig


def sidecar_terminal(execute: Callable, cfg: SidecarConfig) -> Callable:
    return wrap_executor(execute, cfg, output_key="output", rc_key="returncode")


def sidecar_terminal_owned(cfg: SidecarConfig) -> Callable:
    """The sidecar runs the subprocess itself (live progress, byte-exact output)."""
    ex = SidecarExecutor(cfg)
    return lambda command, timeout=None, cwd=None: ex.execute(command, cwd=cwd, timeout=timeout)
