"""Load the committed 29-30 September measurements (rounds 2-3, stage C, stage E) for the report's plots."""

import json
import re
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
LINE = re.compile(r"(?P<pod>\w+) (?P<run>\S+)\s+(?P<case>\S+)\s+l1=(?P<l1>[\d.]+) chunk=(?P<chunk>\d+) (?P<ok>PASS|FAIL) "
                  r"(?P<done>\d+)/(?P<expected>\d+)\s+(?P<elapsed>\d+)s TTFT p50/p95/max (?P<t50>[\d.]+)/(?P<t95>[\d.]+)/"
                  r"(?P<tmax>[\d.]+)s lat p50/p95 (?P<l50>[\d.]+)/(?P<l95>[\d.]+)s hit (?P<hit>[\d.]+) warn (?P<warn>\d+)")
POD_GPU = {"sxm": "H100 SXM", "a100": "A100 SXM", "pro6000": "RTX PRO 6000"}


def round2():
    """Rows from round 2 (A100 repeats and chunks; H100 SXM 24/48 GiB pair), TTFT in seconds."""
    rows = json.loads((RESULTS / "gpu-round2-2026-09-29/results.json").read_text())
    return [dict(gpu=r["gpu"].replace("80GB", "").strip(), run=r["run"], case=r["case"], l1=float(r["l1_gb"]),
                 chunk=int(r["chunk"]), elapsed=r["elapsed_s"], ttft50=r["ttft_ms"]["p50"] / 1e3,
                 lat50=r["latency_ms"]["p50"] / 1e3, hit=r["external_hit_ratio"], warn=r["l1_alloc_warnings_whole_stack"])
            for r in rows if not r["run"].startswith("attempt1")]


def round3():
    """Rows from round 3's labelled summary (one line per replayed root)."""
    rows = []
    for line in (RESULTS / "gpu-round3-2026-09-30/summary.txt").read_text().splitlines():
        m = LINE.match(line)
        if m:
            rows.append(dict(gpu=POD_GPU[m["pod"]], run=m["run"], case=m["case"], l1=float(m["l1"]), chunk=int(m["chunk"]),
                             elapsed=float(m["elapsed"]), ttft50=float(m["t50"]), ttft95=float(m["t95"]),
                             lat50=float(m["l50"]), hit=float(m["hit"]), warn=int(m["warn"])))
    return rows


def retrieval():
    """Stage C medians per GPU: {gpu: {condition: {length: row}}}."""
    out = {}
    for pod, gpu in (("sxm", "H100 SXM"), ("a100", "A100 SXM")):
        rows = json.loads((RESULTS / f"gpu-retrieval-2026-09-30/{pod}-summary.json").read_text())["rows"]
        for r in rows:
            out.setdefault(gpu, {}).setdefault(r["condition"], {})[r["length"]] = r
    return out


def stage_e():
    """Stage E runs per Pod in run order: [(pod, order, step, summary row)]; pod B's first step was contaminated."""
    out = []
    for pod in ("b", "c"):
        rows = json.loads((RESULTS / f"gpu-stage-e-2026-09-30/{pod}-summary.json").read_text())
        out += [(pod.upper(), i, r["step"], r) for i, r in enumerate(rows, 1)]
    return out
