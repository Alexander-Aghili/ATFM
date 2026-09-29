from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ProxyConfig
from .peers import priority_quartile


@dataclass
class CallMeta:
    session_id: str
    cls: str
    tenant: str
    deadline: float | None
    parent: str | None
    turn_index: int
    isl: int
    predicted_osl: int
    t_arrival: float


def estimate_isl(body: dict) -> int:
    """Rough prompt size without a tokenizer: characters / 4."""
    chars = 0
    for m in body.get("messages", []) or []:
        c = m.get("content")
        if isinstance(c, str):
            chars += len(c)
        elif isinstance(c, list):
            chars += sum(len(p.get("text", "")) for p in c if isinstance(p, dict))
    return max(1, chars // 4)


def service_time(meta: CallMeta, cfg: ProxyConfig) -> float:
    return meta.isl / cfg.prefill_tps + meta.predicted_osl / cfg.decode_tps


def weight(cls: str, cfg: ProxyConfig) -> float:
    return cfg.w_interactive if cls == "interactive" else cfg.w_background


def compute_index(meta: CallMeta, cfg: ProxyConfig, e_service_s: float, e_tool_next_s: float) -> float:
    """pi = w(class) * (1 + beta * E[T_tool_next]) / E[S]  (spec 4.3)."""
    return weight(meta.cls, cfg) * (1.0 + cfg.beta * e_tool_next_s) / max(e_service_s, 1e-6)


def tier(meta: CallMeta, cfg: ProxyConfig, now: float, e_service_s: float) -> int:
    """Stable priority tier: 2 interactive under slack, 1 interactive, 0 background."""
    if meta.cls != "interactive":
        return 0
    if meta.deadline is not None and (meta.deadline - now - e_service_s) < cfg.slack_threshold_s:
        return 2
    return 1


def priority_bucket(index: float, tier_indices: list[float]) -> int:
    """Coarse rank quartile of the index among the tier's current indices, 0..3 (3 = highest)."""
    below = int(np.count_nonzero(np.asarray(tier_indices) < index))
    return priority_quartile(below, len(tier_indices))


def promote_at(meta: CallMeta, cfg: ProxyConfig, e_service_s: float) -> float | None:
    """Time at which a waiting interactive call falls under the slack threshold (None if never)."""
    if meta.cls != "interactive" or meta.deadline is None:
        return None
    return meta.deadline - e_service_s - cfg.slack_threshold_s
