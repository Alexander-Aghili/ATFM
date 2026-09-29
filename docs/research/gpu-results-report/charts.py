"""Evidence-backed plots and explicitly hypothetical cache-sizing scenarios."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/research/results/gpu-public-h100-2026-09-29"
OUT = Path(__file__).resolve().parent / "figures"
BLUE, TEAL, ORANGE, INK = "#3066a5", "#168679", "#c77629", "#23364d"
BYTES_PER_TOKEN = 147456
GIB = 2**30


def read(name):
    return json.loads((EVIDENCE / name).read_text())


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": INK, "text.color": INK,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png", dpi=210, bbox_inches="tight", pad_inches=.15)
    plt.close(fig)


def completion():
    fig, ax = plt.subplots(figsize=(9, 2.7), layout="constrained")
    rows = ["Short branch", "Sequential", "Large retry (incomplete)"]
    retained = [21, 24, read("multi-branch-retry/retained-records.json")["retained_request_records"]]
    ax.barh(rows, retained, color=[TEAL, TEAL, ORANGE])
    ax.barh(rows, [0, 0, 27], left=retained, color="#e0e5ec", hatch="//")
    for i, (n, total) in enumerate(zip(retained, [21, 24, 119])):
        ax.text(total + 2, i, f"{n}/{total}", va="center", weight="bold")
    ax.set(xlim=(0, 140), xlabel="Request records", title="MEASURED | Complete roots versus retained partial records")
    ax.invert_yaxis()
    save(fig, "completion")


def latency_panel(ax, profiles, key, title):
    x = np.arange(2)
    for offset, percentile, color in [(-.18, "p50", BLUE), (.18, "p95", TEAL)]:
        values = [p[key][percentile] / 1000 for p in profiles]
        bars = ax.bar(x + offset, values, width=.34, color=color, label=percentile)
        ax.bar_label(bars, fmt="%.2f", padding=3)
    ax.set(xticks=x, xticklabels=["Short branch", "Sequential"], ylabel="Seconds", title=title)
    ax.margins(y=.25)
    ax.legend(frameon=False, loc="upper left")


def latency():
    profiles = [read(f"{name}/profile.json") for name in ["short-branch", "sequential"]]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), layout="constrained")
    latency_panel(axes[0], profiles, "time_to_first_token", "MEASURED | Time to first token")
    latency_panel(axes[1], profiles, "request_latency", "MEASURED | Full response")
    save(fig, "latency")


def reuse():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    names = ["Short branch", "Sequential"]
    hits = [714560 / 820262 * 100, 1466560 / 1560190 * 100]
    bars = axes[0].bar(names, hits, color=TEAL)
    axes[0].bar_label(bars, fmt="%.2f%%", padding=3)
    axes[0].set(ylim=(0, 110), ylabel="Hit / query tokens (%)", title="MEASURED | External reuse")
    bars = axes[1].bar(names, np.array([16529227776, 19329712128]) / GIB, color=BLUE)
    axes[1].bar_label(bars, fmt="%.2f", padding=3)
    axes[1].axhline(19.2, color=ORANGE, ls="--", label="80% watermark")
    axes[1].axhline(24, color=INK, label="Hard cap")
    axes[1].set(ylim=(0, 29), ylabel="GiB", title="MEASURED | Final L1 occupancy")
    axes[1].legend(frameon=False, fontsize=8, loc="upper left")
    save(fig, "reuse")


def tools_chart():
    groups = read("tools-summary.json")["results"]
    counts = [sum(r["passed"] for r in rows) for rows in groups.values()]
    fig, ax = plt.subplots(figsize=(9, 3), layout="constrained")
    labels = ["Single function", "Function selection", "Parallel", "Parallel + selection"]
    ax.barh(labels, counts, color=TEAL)
    ax.barh(labels, 10 - np.array(counts), left=counts, color=ORANGE)
    for i, count in enumerate(counts):
        ax.text(10.2, i, f"{count}/10", va="center")
    ax.set(xlim=(0, 11.4), xlabel="Structural checks", title="MEASURED | BFCL-derived sample: 39/40 pass")
    ax.invert_yaxis()
    save(fig, "tools")


def capacity():
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    tokens = np.linspace(0, 131072, 200)
    for count, color in [(1, BLUE), (2, TEAL), (4, ORANGE)]:
        ax.plot(tokens / 1000, tokens * count * BYTES_PER_TOKEN / GIB,
                label=f"{count} independent context(s)", color=color, lw=2.2)
    ax.axhline(24, color=INK, label="24 GiB hard cap")
    ax.axhline(19.2, color=INK, ls="--", label="19.2 GiB watermark")
    ax.scatter([100], [100000 * 2 * BYTES_PER_TOKEN / GIB], color=TEAL, zorder=3)
    ax.annotate("Two 100k contexts: 27.47 GiB", (100, 27.47), (56, 38), arrowprops={"arrowstyle": "->"})
    ax.set(xlabel="Tokens per context (thousands)", ylabel="KV tensor memory (GiB)",
           title="DERIVED | Linear KV growth crosses the CPU tier", xlim=(0, 132), ylim=(0, 76))
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    save(fig, "capacity")


def sharing_heatmap(ax):
    sizes, counts = np.array([16000, 32000, 64000, 100000, 131072]), np.arange(1, 5)
    values = counts[:, None] * sizes * BYTES_PER_TOKEN / GIB
    ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0, vmax=72)
    for i in range(4):
        for j in range(5):
            ax.text(j, i, f"{values[i,j]:.1f}", ha="center", va="center",
                    color="white" if values[i,j] > 40 else INK, fontsize=9)
    ax.set(xticks=range(5), xticklabels=["16k", "32k", "64k", "100k", "131k"],
           yticks=range(4), yticklabels=counts, xlabel="Tokens per independent context",
           ylabel="Resident context count", title="DERIVED | Tensor GiB")


def sharing():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), layout="constrained")
    sharing_heatmap(axes[0])
    prefix = np.linspace(0, 100000, 100)
    axes[1].plot(prefix / 1000, (200000 - prefix) * BYTES_PER_TOKEN / GIB, color=TEAL, lw=2.5)
    axes[1].axhline(24, color=INK, label="Hard cap")
    axes[1].axhline(19.2, color=ORANGE, ls="--", label="Watermark")
    axes[1].set(xlabel="Shared prefix (thousands of tokens)", ylabel="Unique tensor GiB",
                title="DERIVED | Two 100k contexts", ylim=(10, 30))
    axes[1].legend(frameon=False, fontsize=8)
    save(fig, "sharing")


def chunks():
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    sizes = np.array([16, 64, 256])
    values = [np.ceil(100000 / sizes), sizes * BYTES_PER_TOKEN / 2**20]
    for ax, value, title, unit in zip(axes, values, ["Objects for 100k tokens", "Bytes per full chunk"], ["Objects", "MiB"]):
        bars = ax.bar([str(s) for s in sizes], value, color=[BLUE, TEAL, ORANGE])
        ax.bar_label(bars, fmt="%g", padding=3)
        ax.set(xlabel="Tokens per chunk", ylabel=unit, title=f"DERIVED | {title}")
        ax.margins(y=.2)
    save(fig, "chunks")


def transfer():
    fig, ax = plt.subplots(figsize=(9, 3.6), layout="constrained")
    tokens = np.linspace(1000, 131072, 200)
    for speed, color in zip([1, 5, 20, 40], [ORANGE, INK, BLUE, TEAL]):
        ax.plot(tokens / 1000, tokens * BYTES_PER_TOKEN / GIB / speed,
                label=f"{speed} GiB/s assumed", color=color, lw=2)
    ax.axvline(100, color="#9ca9b8", ls=":")
    ax.set(yscale="log", xlabel="Cached tokens (thousands)", ylabel="One-hop seconds (log scale)",
           title="ILLUSTRATIVE | Transfer lower bound; bandwidth is not measured")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.grid(axis="y", alpha=.2)
    save(fig, "transfer")


def prefetch():
    fig, ax = plt.subplots(figsize=(9, 3.3), layout="constrained")
    for i, start in enumerate([10, 9, 6, 0]):
        ax.broken_barh([(start, 3)], (i - .3, .6), facecolors=BLUE)
        if start + 3 < 10:
            ax.broken_barh([(start + 3, 10 - start - 3)], (i - .3, .6), facecolors="#bddbd5")
        if start + 3 > 10:
            ax.broken_barh([(10, start + 3 - 10)], (i - .3, .6), facecolors=ORANGE)
    ax.axvline(10, color=INK, ls="--", label="Request returns")
    ax.set(yticks=range(4), yticklabels=["On demand: 3 s exposed", "Late: 2 s exposed", "On time: 1 s holding", "Early: 7 s holding"],
           xlabel="Illustrative time (s)", xlim=(-.3, 14), title="ILLUSTRATIVE | Loading, waiting, and exposed delay")
    ax.invert_yaxis()
    ax.plot([], [], color=BLUE, lw=7, label="Background load")
    ax.plot([], [], color=ORANGE, lw=7, label="Exposed load")
    ax.plot([], [], color="#bddbd5", lw=7, label="Resident waiting")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(.5, -.22))
    save(fig, "prefetch")


def build():
    setup()
    for chart in [completion, latency, reuse, tools_chart, capacity, sharing, chunks, transfer, prefetch]:
        chart()


if __name__ == "__main__":
    build()
