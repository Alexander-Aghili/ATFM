"""Measured plots for rounds 2-3 and stage C, read from committed evidence (see evidence.py)."""

import matplotlib.pyplot as plt
import numpy as np

import evidence
from charts import BLUE, INK, ORANGE, TEAL, save

GREY = "#9ca9b8"


def capacity_measured():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    rows = [r for r in evidence.round3() if r["case"] == "multi-branch"]
    for ax, key, label in zip(axes, ["elapsed", "ttft50"], ["Replay elapsed (s)", "Median TTFT (s)"]):
        for gpu, dx in (("H100 SXM", -.12), ("A100 SXM", .12)):
            for l1, x, color in ((24.0, 0, ORANGE), (48.0, 1, TEAL)):
                vals = [r[key] for r in rows if r["gpu"] == gpu and r["l1"] == l1]
                ax.scatter([x + dx] * len(vals), vals, s=46, color=color, marker="o" if gpu == "H100 SXM" else "s", zorder=3)
        ax.set(xticks=[0, 1], xticklabels=["24 GiB L1", "48 GiB L1"], ylabel=label, xlim=(-.5, 1.5))
        ax.grid(axis="y", alpha=.2)
    axes[0].set_title("MEASURED | 119-request root, round 3")
    axes[1].scatter([], [], color=GREY, marker="o", label="H100 SXM (2 runs each)")
    axes[1].scatter([], [], color=GREY, marker="s", label="A100 SXM (1 run each)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper right")
    save(fig, "capacity-measured")


def chunk_points():
    """(panel, pod label, marker) -> {chunk: [median TTFT per run]}; pods are never pooled."""
    pods = ((("A100", "round 2 Pod", "o"), evidence.round2(), "A100"),
            (("A100", "round 3 Pod", "s"), evidence.round3(), "A100"),
            (("RTX PRO 6000", "round 3 Pod", "^"), evidence.round3(), "RTX PRO 6000"))
    out = {}
    for key, rows, gpu in pods:
        seq = [r for r in rows if r["case"] == "sequential" and r["l1"] == 24 and r["gpu"].startswith(gpu)]
        out[key] = {c: [r["ttft50"] for r in seq if r["chunk"] == c] for c in (16, 64, 256)}
    return out


def chunk_measured():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), layout="constrained")
    panels = {"A100": axes[0], "RTX PRO 6000": axes[1]}
    for (gpu, pod, marker), points in chunk_points().items():
        ax = panels[gpu]
        for i, (chunk, vals) in enumerate(points.items()):
            ax.scatter([i] * len(vals), vals, s=40, color=[ORANGE, BLUE, TEAL][i], marker=marker, zorder=3,
                       label=pod if i == 0 else None)
    for gpu, ax in panels.items():
        ax.set(xticks=range(3), xticklabels=["16", "64", "256"], xlabel="Tokens per chunk",
               ylabel="Median TTFT per run (s)", title=f"MEASURED | {gpu}, sequential root")
        ax.grid(axis="y", alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    save(fig, "chunks-measured")


def retrieval_chart():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), layout="constrained", sharey=True)
    series = (("cold", "Cold recompute", ORANGE, "o"), ("l2", "L2 on demand", GREY, "s"), ("l1", "Warmed L1", TEAL, "^"))
    for ax, (gpu, cond) in zip(axes, evidence.retrieval().items()):
        for key, label, color, marker in series:
            lengths = sorted(cond[key])
            ax.plot([n / 1000 for n in lengths], [cond[key][n]["client_s_p50"] for n in lengths],
                    marker=marker, color=color, lw=2, label=label)
        warm = sorted(cond["l1"])
        ax.plot([n / 1000 for n in warm], [cond["l1"][n]["prefetch_s_p50"] for n in warm], ls=":", color=INK, label="Warm time (off path)")
        ax.set(xscale="log", yscale="log", xlabel="Context tokens", title=f"MEASURED | {gpu}")
        ax.set_xticks([2.048, 8.192, 32.768, 98.304], ["2k", "8k", "33k", "98k"])
        ax.minorticks_off()
        ax.grid(alpha=.2, which="both")
    axes[0].set_ylabel("Median seconds (log)")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "retrieval")


def gpus_chart():
    fig, ax = plt.subplots(figsize=(9, 3.0), layout="constrained")
    a100 = [r["elapsed"] for r in evidence.round2() if r["case"] == "sequential" and r["run"].startswith("rep")]
    pro = [r["elapsed"] for r in evidence.round3() if r["case"] == "sequential" and r["gpu"] == "RTX PRO 6000" and r["chunk"] == 16]
    for i, (vals, color) in enumerate([([305], ORANGE), (a100, BLUE), (pro, TEAL)]):
        ax.scatter(vals, [i] * len(vals), s=40, color=color, zorder=3)
    ax.set(yticks=range(3), yticklabels=["H100 NVL (29 Sep, 1 run)", "A100 SXM (round 2, 3 runs)", "RTX PRO 6000 (round 3, 3 runs)"],
           xlabel="Sequential root replay elapsed (s)", title="MEASURED | Same root, 16-token chunks, one Pod per GPU")
    ax.grid(axis="x", alpha=.2)
    save(fig, "gpus")


ARM_COLORS = {"direct": INK, "proxy": GREY, "atfm-q10": ORANGE, "atfm-q50": TEAL}


def stage_e_panel(ax, key, label):
    for pod, x0 in (("B", 0), ("C", 5)):
        runs = [(i, step, row) for p, i, step, row in evidence.stage_e() if p == pod]
        xs = [x0 + i for i, _, _ in runs]
        ax.plot(xs, [row[key]["p50"] if isinstance(row[key], dict) else row[key] for _, _, row in runs], color="#d5dbe2", zorder=1)
        for i, step, row in runs:
            value = row[key]["p50"] if isinstance(row[key], dict) else row[key]
            contaminated = pod == "B" and i == 1
            ax.scatter(x0 + i, value, s=60, zorder=3, color="white" if contaminated else ARM_COLORS[row["arm"]],
                       edgecolor=ARM_COLORS[row["arm"]], linewidth=1.8)
    ax.set_xticks([1, 2, 3, 4, 6, 7, 8, 9], ["B1", "B2", "B3", "B4", "C1", "C2", "C3", "C4"])
    ax.set(xlabel="Pod and run order", ylabel=label)
    ax.grid(axis="y", alpha=.2)


def stage_e_chart():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    stage_e_panel(axes[0], "elapsed_s", "Replay elapsed (s)")
    stage_e_panel(axes[1], "ttft_s", "Median TTFT (s)")
    axes[0].set_title("MEASURED | Stage E, 119-request root, 24 GiB")
    for arm, color in ARM_COLORS.items():
        axes[1].scatter([], [], color=color, s=40, label=arm)
    axes[1].scatter([], [], color="white", edgecolor=INK, s=40, label="shared host (excluded)")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="outside lower center", ncol=5)
    save(fig, "stage-e")


def build():
    for chart in [capacity_measured, chunk_measured, retrieval_chart, gpus_chart, stage_e_chart]:
        chart()
