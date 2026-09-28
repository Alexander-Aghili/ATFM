"""Worker metrics from a Prometheus text page (Dynamo backend / vLLM worker exporters) into
`worker.metrics` events, the board's capacity input (spec 5.1, D1 metrics scrape)."""
from __future__ import annotations

import re

from atfm.schema.events import WorkerMetrics

_LINE = re.compile(r'^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+([-+0-9.eE]+|NaN|\+Inf|-Inf)\s*$')
_LABEL = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:[^"\\]|\\.)*)"')
_WORKER_LABELS = ("instance", "worker", "worker_id", "pod", "endpoint")


def parse_prometheus(text: str) -> list[tuple[str, dict[str, str], float]]:
    rows = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        m = _LINE.match(line)
        if not m:
            continue
        labels = {k: v for k, v in _LABEL.findall(m.group(2) or "")}
        try:
            rows.append((m.group(1), labels, float(m.group(3))))
        except ValueError:
            continue
    return rows


def _worker_of(labels: dict[str, str], default: str | None) -> str | None:
    """Prometheus-server labels (instance, pod, ...) name the worker; a Dynamo page names component and
    endpoint; a worker's own page has neither and takes the scrape's `default` (its URL or host)."""
    if "dynamo_component" in labels or "dynamo_endpoint" in labels:
        return "/".join(v for v in (labels.get("dynamo_component"), labels.get("dynamo_endpoint")) if v)
    for k in _WORKER_LABELS:
        if k in labels:
            return labels[k]
    return default


def worker_metrics_from_prometheus(text: str, t: float, default_total_blocks: int | None = None,
                                   default_worker_id: str | None = None) -> list[WorkerMetrics]:
    """One event per worker. Total and used blocks come from `*kv_total_blocks` / `*kv_active_blocks`
    (Dynamo); a vLLM-only page gives `gpu_cache_usage_perc`, converted with `default_total_blocks`.
    Queue depth is `num_requests_waiting` or `requests_pending`. Workers without a total are skipped."""
    per = _group_metrics(text, default_worker_id)
    out = []
    for w, d in per.items():
        total = d.get("total", default_total_blocks)
        if total is None:
            continue
        used = d.get("used")
        if used is None:
            if "frac" not in d:
                continue
            used = int(round(d["frac"] * total))
        out.append(WorkerMetrics(t=t, worker_id=w, kv_blocks_used=used, kv_blocks_total=int(total), queue_depth=d.get("queue", 0)))
    return out


def _group_metrics(text, default_worker_id):
    per: dict[str, dict] = {}
    for name, labels, value in parse_prometheus(text):
        w = _worker_of(labels, default_worker_id)
        if w is None:
            continue
        d = per.setdefault(w, {})
        if name.endswith("kv_total_blocks"):
            d["total"] = int(round(value))
        elif name.endswith("kv_active_blocks") or name.endswith("kv_blocks_used"):
            d["used"] = int(round(value))
        elif name.endswith("gpu_cache_usage_perc") or name.endswith("kv_cache_usage_perc"):
            d["frac"] = float(value)
        elif name.endswith("num_requests_waiting") or name.endswith("requests_pending") or name.endswith("queue_depth"):
            d["queue"] = int(round(value))
    return per
