"""Harbor adapter (spec 7): wrap `BaseAgent.environment.exec(cmd, timeout=...) -> (stdout, exit_code)`."""
from __future__ import annotations

from typing import Callable

from atfm.sidecar.adapters import wrap_executor
from atfm.sidecar.config import SidecarConfig


def sidecar_exec(exec_fn: Callable, cfg: SidecarConfig) -> Callable:
    return wrap_executor(exec_fn, cfg, tuple_result=True)
