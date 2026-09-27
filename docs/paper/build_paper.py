"""Build docs/paper/atfm-paper.html (artifact + print source) from the reladraw figures and the results.

    uv run python docs/paper/build_paper.py
    google-chrome --headless=new --no-pdf-header-footer --print-to-pdf=docs/paper/atfm-paper.pdf file://.../atfm-paper.html
"""
from __future__ import annotations

import base64
import math
from pathlib import Path

HERE = Path(__file__).parent
FIG = HERE / "fig"


def fig_uri(name: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode((FIG / f"{name}.svg").read_bytes()).decode()


# ----------------------------------------------------------------------------- data
HORIZONS = [10, 30, 120, 300, 900]
TRACELAB = {100: [1.08, 1.25, 1.63, 1.96, 3.00], 200: [1.03, 1.50, 2.11, 2.53, 3.96], 400: [1.01, 1.70, 2.55, 3.31, 5.09]}
AGENTX = {100: [1.10, 1.39, 2.20, 2.97, 2.73], 200: [1.21, 1.64, 3.11, 4.27, 3.34], 400: [1.49, 1.97, 3.69, 5.43, 3.58]}
H1B = [("fmt build -j1", 52.6, 47.2, 7.7), ("fmt build -j2", 8.7, 10.4, 15.6), ("pipeline linear a", 26.2, 26.3, 0.5),
       ("pipeline linear b", 14.3, 16.2, 1.4), ("pipeline linear c", 21.3, 21.4, 0.8), ("pipeline linear d", 8.9, 11.5, 1.7),
       ("pipeline staged a", 25.0, 24.9, 4.8), ("pipeline staged b", 69.0, 39.2, 19.8), ("numpy suite (unseen)", 4.0, 14.0, 14.6),
       ("all phases", 30.1, 24.9, 8.9)]
H2 = [("proxy rules", 0.014, -0.002, 0.028, -35, -90, 17), ("forecast M1", 0.088, 0.074, 0.101, 790, 736, 842),
      ("forecast M2", 0.093, 0.080, 0.105, 789, 739, 837), ("M2, no holds", -0.013, -0.028, 0.001, -35, -97, 10),
      ("oracle (heap)", 0.070, 0.054, 0.083, 1141, 1090, 1190), ("working set", 0.076, 0.063, 0.089, 2082, 2035, 2136)]


# ----------------------------------------------------------------------------- charts (hand-placed SVG, one scale each)
def chart_ratio(data: dict, title: str) -> str:
    W, H, L, R, T, B = 460, 300, 52, 16, 28, 44
    xs = [math.log10(h) for h in HORIZONS]
    x0, x1 = xs[0], xs[-1]
    ymax = 6.0
    def X(v): return L + (v - x0) / (x1 - x0) * (W - L - R)
    def Y(v): return T + (1 - v / ymax) * (H - T - B)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{title}: ratio of Kalman baseline loss to elapsed-time model loss versus horizon at three arrival rates">']
    out.append(f'<text x="{L}" y="16" class="ct">{title}</text>')
    for g in [1, 2, 3, 4, 5, 6]:
        out.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(g):.1f}" y2="{Y(g):.1f}" class="grid"/>')
        out.append(f'<text x="{L-6}" y="{Y(g)+4:.1f}" class="tick" text-anchor="end">{g}x</text>')
    out.append(f'<line x1="{L}" x2="{W-R}" y1="{Y(1):.1f}" y2="{Y(1):.1f}" class="base"/>')
    for h, x in zip(HORIZONS, xs):
        lab = {10: "10 s", 30: "30 s", 120: "2 min", 300: "5 min", 900: "15 min"}[h]
        out.append(f'<text x="{X(x):.1f}" y="{H-B+18}" class="tick" text-anchor="middle">{lab}</text>')
    out.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-6}" class="axis" text-anchor="middle">forecast horizon</text>')
    for i, (rate, ys) in enumerate(sorted(data.items())):
        pts = " ".join(f"{X(x):.1f},{Y(y):.1f}" for x, y in zip(xs, ys))
        out.append(f'<polyline points="{pts}" class="s{i+1} line"/>')
        for x, y in zip(xs, ys):
            out.append(f'<circle cx="{X(x):.1f}" cy="{Y(y):.1f}" r="4" class="s{i+1} dot"/>')
        out.append(f'<text x="{X(xs[-1])+6:.1f}" y="{Y(ys[-1])+4:.1f}" class="lab">{rate}/h</text>' if False else "")
    # legend (fixed order, same for both panels)
    for i, rate in enumerate(sorted(data)):
        out.append(f'<line x1="{L+8+i*110}" x2="{L+30+i*110}" y1="{T-8}" y2="{T-8}" class="s{i+1} line"/>'
                   f'<text x="{L+36+i*110}" y="{T-4}" class="tick">{rate} sessions/h</text>')
    out.append("</svg>")
    return "".join(out)


def chart_h1b() -> str:
    rows = H1B
    W, L, R, T = 720, 150, 16, 26
    rh, gap = 30, 10
    H = T + len(rows) * (rh + gap) + 34
    vmax = 70.0
    def X(v): return L + v / vmax * (W - L - R)
    out = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="H1b: mean q90 pinball loss on remaining seconds per job family for B2, M1 and M2, lower is better">']
    for g in range(0, 71, 10):
        out.append(f'<line x1="{X(g):.1f}" x2="{X(g):.1f}" y1="{T-4}" y2="{H-30}" class="grid"/>')
        out.append(f'<text x="{X(g):.1f}" y="{H-14}" class="tick" text-anchor="middle">{g}</text>')
    out.append(f'<text x="{(L+W-R)/2:.1f}" y="{H-2}" class="axis" text-anchor="middle">mean q90 pinball loss on remaining seconds (lower is better)</text>')
    bh = (rh - 4) / 3
    for r, (name, b2, m1, m2) in enumerate(rows):
        y = T + r * (rh + gap)
        cls = "lab strong" if name == "all phases" else "lab"
        out.append(f'<text x="{L-8}" y="{y+rh/2+4:.1f}" class="{cls}" text-anchor="end">{name}</text>')
        for i, v in enumerate((b2, m1, m2)):
            yy = y + i * (bh + 2)
            out.append(f'<rect x="{L}" y="{yy:.1f}" width="{max(X(v)-L,1):.1f}" height="{bh:.1f}" rx="2" class="s{i+1} bar"/>')
            out.append(f'<text x="{X(v)+4:.1f}" y="{yy+bh-1:.1f}" class="val">{v:g}</text>')
    for i, n in enumerate(("B2 history, elapsed ignored", "M1 elapsed time", "M2 live progress")):
        out.append(f'<rect x="{L+8+i*215}" y="{T-22}" width="12" height="10" rx="2" class="s{i+1} bar"/><text x="{L+24+i*215}" y="{T-13}" class="tick">{n}</text>')
    out.append("</svg>")
    return "".join(out)


def chart_h2() -> str:
    def panel(x_off, title, vals, vmin, vmax, fmt, ticks):
        W, L, R, T = 360, 118, 14, 30
        rh, gap = 26, 8
        H = T + len(H2) * (rh + gap) + 30
        def X(v): return L + (v - vmin) / (vmax - vmin) * (W - L - R)
        o = [f'<g transform="translate({x_off},0)">', f'<text x="{L}" y="14" class="ct">{title}</text>']
        for g in ticks:
            o.append(f'<line x1="{X(g):.1f}" x2="{X(g):.1f}" y1="{T-4}" y2="{H-26}" class="{"base" if g == 0 else "grid"}"/>')
            o.append(f'<text x="{X(g):.1f}" y="{H-12}" class="tick" text-anchor="middle">{fmt(g)}</text>')
        for r, (name, m, lo, hi) in enumerate(vals):
            y = T + r * (rh + gap) + rh / 2
            o.append(f'<text x="{L-8}" y="{y+4:.1f}" class="lab" text-anchor="end">{name}</text>')
            o.append(f'<line x1="{X(lo):.1f}" x2="{X(hi):.1f}" y1="{y:.1f}" y2="{y:.1f}" class="ci"/>')
            o.append(f'<circle cx="{X(m):.1f}" cy="{y:.1f}" r="5" class="s1 dot"/>')
        o.append("</g>")
        return "".join(o), H
    a, H = panel(0, "interactive SLO attainment, difference vs native", [(n, m, lo, hi) for n, m, lo, hi, *_ in H2], -0.04, 0.12,
                 lambda g: f"{g:+.2f}", [-0.04, 0, 0.04, 0.08, 0.12])
    b, _ = panel(370, "background job completion time, difference vs native (s)", [(n, j, lo, hi) for n, _, _, _, j, lo, hi in H2], -200, 2200,
                 lambda g: f"{g:+d}", [0, 500, 1000, 1500, 2000])
    return (f'<svg viewBox="0 0 740 {H}" role="img" aria-label="H2 loaded regime: paired differences against the native arm with 95% bootstrap intervals, interactive SLO on the left and background completion time on the right">'
            + a + b + "</svg>")


# ----------------------------------------------------------------------------- page
CSS = """
:root{color-scheme:light;--bg:#fbfaf7;--paper:#ffffff;--ink:#1a1916;--ink2:#4b4945;--mute:#7d7a73;--rule:#dcd9d1;--accent:#5b46b0;--accent2:#c9531f;
--plate:#ffffff;--code:#f1efe9;--s1:#2a78d6;--s2:#eb6834;--s3:#1baf7a;--good:#1d7a3c;--warn:#b8860b}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--bg:#16161a;--paper:#1d1d22;--ink:#ecebe6;--ink2:#c5c3bb;--mute:#8f8d86;--rule:#33333a;--accent:#a596e8;--accent2:#e8865a;
--plate:#f7f6f2;--code:#26262c;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#4fbf74;--warn:#d9a52c}}
:root[data-theme="dark"]{color-scheme:dark;--bg:#16161a;--paper:#1d1d22;--ink:#ecebe6;--ink2:#c5c3bb;--mute:#8f8d86;--rule:#33333a;--accent:#a596e8;--accent2:#e8865a;
--plate:#f7f6f2;--code:#26262c;--s1:#3987e5;--s2:#d95926;--s3:#199e70;--good:#4fbf74;--warn:#d9a52c}
body{background:var(--bg);color:var(--ink);font-family:"Source Serif 4",Georgia,"Times New Roman",serif;font-size:15.5px;line-height:1.55;margin:0;padding-block:0 48px;padding-inline:16px}
.sheet{max-width:860px;margin:0 auto;background:var(--paper);padding:48px clamp(16px,5vw,64px) 64px;border:1px solid var(--rule)}
h1,h2,h3,h4{font-family:"Source Serif 4",Georgia,serif;text-wrap:balance;line-height:1.2;margin:0}
h1{font-size:2.1rem;font-weight:700;letter-spacing:-.01em}
h2{font-size:1.35rem;font-weight:700;margin-top:2.4rem;padding-top:.6rem;border-top:2px solid var(--ink)}
h3{font-size:1.08rem;font-weight:700;margin-top:1.6rem}
h4{font-size:1rem;font-weight:700;margin-top:1.1rem;font-style:italic}
p{margin:.7rem 0;max-width:68ch}
.meta{font-family:"IBM Plex Sans",system-ui,sans-serif;color:var(--ink2);font-size:.92rem;margin:.6rem 0 0}
.eyebrow{font-family:"IBM Plex Sans",system-ui,sans-serif;text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;color:var(--accent);font-weight:600}
.abstract{border-left:3px solid var(--accent);padding:.2rem 0 .2rem 1.1rem;margin:1.6rem 0;color:var(--ink2)}
.abstract p{max-width:none}
figure{margin:1.4rem 0}
figure .plate{background:var(--plate);border:1px solid var(--rule);padding:12px;overflow-x:auto}
figure img{display:block;max-width:100%;height:auto;margin:0 auto}
figcaption{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.86rem;color:var(--ink2);margin-top:.5rem;line-height:1.45}
figcaption b{color:var(--ink)}
.tblwrap{overflow-x:auto;margin:1rem 0}
table{border-collapse:collapse;font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.84rem;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:.38rem .55rem;border-bottom:1px solid var(--rule);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--ink2);border-bottom:1.5px solid var(--ink);font-size:.78rem;text-transform:uppercase;letter-spacing:.05em}
td.n,th.n{text-align:right}
td b{color:var(--ink)}
caption{caption-side:top;text-align:left;font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.86rem;color:var(--ink2);padding:.3rem 0 .5rem}
caption b{color:var(--ink)}
code,pre{font-family:"IBM Plex Mono",ui-monospace,Menlo,monospace;font-size:.84em}
code{background:var(--code);padding:.05em .3em;border-radius:3px}
pre{background:var(--code);padding:.8rem 1rem;overflow-x:auto;line-height:1.45;border-radius:4px}
ul,ol{padding-left:1.3rem;max-width:70ch}li{margin:.3rem 0}
.callout{border:1px solid var(--rule);border-left:3px solid var(--accent2);padding:.6rem 1rem;margin:1.2rem 0;font-size:.95rem;background:color-mix(in srgb,var(--paper) 92%,var(--accent2))}
.callout p{max-width:none;margin:.35rem 0}
.two{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media (max-width:640px){.two{grid-template-columns:1fr}}
.chart svg{width:100%;height:auto;display:block;font-family:"IBM Plex Sans",system-ui,sans-serif}
.chart .grid{stroke:var(--rule);stroke-width:1}.chart .base{stroke:var(--ink2);stroke-width:1.2}
.chart .tick{fill:var(--mute);font-size:11px}.chart .axis{fill:var(--ink2);font-size:11px}.chart .lab{fill:var(--ink2);font-size:11.5px}.chart .lab.strong{fill:var(--ink);font-weight:600}
.chart .ct{fill:var(--ink);font-size:12px;font-weight:600}.chart .val{fill:var(--ink2);font-size:10px}
.chart .line{fill:none;stroke-width:2;stroke-linejoin:round}.chart .dot{stroke:var(--paper);stroke-width:2}.chart .bar{stroke:none}.chart .ci{stroke:var(--ink2);stroke-width:2}
.chart .s1.line,.chart .s1.ci{stroke:var(--s1)}.chart .s2.line{stroke:var(--s2)}.chart .s3.line{stroke:var(--s3)}
.chart .s1.dot,.chart .s1.bar{fill:var(--s1)}.chart .s2.dot,.chart .s2.bar{fill:var(--s2)}.chart .s3.dot,.chart .s3.bar{fill:var(--s3)}
.status{display:inline-block;font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.72rem;font-weight:600;letter-spacing:.04em;padding:.1em .5em;border-radius:3px;border:1px solid currentColor}
.done{color:var(--good)}.part{color:var(--warn)}.next{color:var(--mute)}
.toc{font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:.9rem;columns:2;column-gap:2rem;margin:1rem 0 0}
.toc a{color:var(--ink2);text-decoration:none}.toc a:hover{color:var(--accent)}
a{color:var(--accent)}
.refs li{font-size:.9rem;margin:.4rem 0}
.foot{font-family:"IBM Plex Sans",system-ui,sans-serif;color:var(--mute);font-size:.8rem;margin-top:3rem;border-top:1px solid var(--rule);padding-top:.8rem}
@media print{
 :root{--bg:#fff;--paper:#fff}
 @page{size:A4;margin:16mm 15mm}
 body{padding:0;font-size:10.5pt;background:#fff}
 .sheet{max-width:none;border:none;padding:0}
 h2{break-after:avoid}h3,h4{break-after:avoid}
 figure,table,.callout,.two{break-inside:avoid}
 .toc{display:none}
 a{color:inherit;text-decoration:none}
}
"""


def html() -> str:
    f = fig_uri
    return f"""<title>ATFM Technical Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;0,8..60,700;1,8..60,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>{CSS}</style>
<div class="sheet">
<div class="eyebrow">Technical report · working draft · 2026-09-27</div>
<h1 style="margin-top:.4rem">Agent Traffic Flow Management: forecasting and shaping LLM demand from in-flight agent sessions on a shared GPU pool</h1>
<p class="meta">Alexander Aghili. Build 3fbd02b + feature branch of 2026-09-27; 151 tests; simulator and forecast results as of this date. Prepared with Claude Code.</p>

<div class="abstract">
<p><b>Abstract.</b> Fleets of LLM agents alternate between short model calls and tool executions that last from milliseconds to hours. On a shared serving pool this produces demand that is bursty, heavy-tailed and, we argue, largely <i>predictable</i>: most of what arrives in the next few minutes is already in flight, running a tool whose remaining time can be estimated. ATFM (Agent Traffic Flow Management) treats the pool the way air-traffic flow management treats airspace: a demand board forecasts KV-cache and prefill demand per horizon from the live state of every session; a proxy admits, orders and, for deferrable work, holds calls against that forecast; and every hold is charged its measured cost. This report describes the design (twelve recorded decisions), the implementation (a forecast core, laptop plumbing on NVIDIA Dynamo, and a closed-loop fleet simulator; 4.6 kLOC of Python, 151 tests), and the evidence to date. On two public corpora of real agent sessions (TraceLab, 4,265 sessions; AgentX, 393 Claude Code sessions with sub-agent fan-out), conditioning on in-flight state beats a Kalman history baseline by 3 to 5x on q90 pinball loss at 5 to 15 minute horizons and is never worse at 10 s (H1a). On 202 observation points inside real long tools (cmake builds, data pipelines, the numpy test suite), live progress cuts remaining-time error 2.8x over elapsed time alone (H1b). In the simulator, under load, forecast-driven holds raise interactive SLO attainment by 9 points over engine-native priority at a 24% background completion-time cost, a no-hold ablation shows the holds are what buy the tail, and a ThunderAgent-style pause baseline reaches the same SLO at 2.6x the background cost (H2, synthetic, ordering only). We state what remains unvalidated, chiefly the KV model, and give the design of the two-GPU study that would validate it.</p>
</div>

<nav class="toc">
<a href="#s1">1. Problem and claim</a><br><a href="#s2">2. Related systems</a><br><a href="#s3">3. Design</a><br><a href="#s4">4. Implementation</a><br>
<a href="#s5">5. Evaluation method</a><br><a href="#s6">6. Results</a><br><a href="#s7">7. Limitations</a><br><a href="#s8">8. Next steps</a><br><a href="#refs">References</a>
</nav>

<h2 id="s1">1. Problem and claim</h2>
<p>An agent session is a loop: the harness sends a prompt, the model streams a reply, the reply launches a tool (a shell command, a test suite, a build, a query, a sub-agent), the tool's output becomes the next prompt. Between model calls the session holds no GPU compute but its context, and therefore its KV cache, is still valuable: the next call reuses it as prefix. When many sessions share a pool, the serving system sees a stream of calls whose timing is set elsewhere, by tools running on CPUs, CI runners and remote services.</p>
<p>Today's serving stacks treat each call as independent. Priority fields, prefix-aware routing and KV eviction policies react to what has arrived. Yet the sessions that will arrive in the next thirty seconds to fifteen minutes are, for the most part, already known: they are the ones whose tool started a while ago. Their return time is uncertain but not unknowable, and their context size is known exactly. The central claim of this program is that this in-flight information is worth acting on, and the three hypotheses below make it falsifiable.</p>
<ul>
<li><b>H1a (demand board).</b> In-flight session state (phase, elapsed time, context size) forecasts fleet KV and prefill demand at 30 s to 15 min better than history-based predictors of the same series.</li>
<li><b>H1b (sidecar).</b> Live tool progress improves the forecast beyond elapsed time, measurable only on long-tool workloads.</li>
<li><b>H2 (controller).</b> A proxy that admits, orders and holds calls against the forecast improves the interactive tail at a stated and measured cost to deferrable work, in closed loop.</li>
</ul>
<p>The claims are deliberately separable. A demand board can be right and useless if the controller cannot act on it; a controller can help by rules alone with no forecast. The evaluation therefore scores forecasts on their own terms (proper scoring rules against realized demand) and policies only in closed loop, where agents react to the reply times they receive.</p>

<div class="callout"><p><b>Production evidence for the workload shape.</b> DeepSeek's DSec report (arXiv 2609.22978, September 2026) measures the CPU side of this workload at scale: about 380,000 concurrent sandboxes and 3 million a day per unit; median sandbox lifetime 15 to 17 minutes with a p99 over three hours; about 90% of sandboxes use at most 5% of their requested CPU on average, the tool phase being "short CPU bursts separated by periods" of waiting on the model; and from DeepSeek-V4.1 the agent loop runs off the GPU pods and calls model serving as a separate service. That is exactly the regime this design assumes: long-lived, mostly idle sessions with sparse, bursty GPU demand, decoupled from the pool. The report says nothing about inference serving or forecasting; it is evidence for the demand's shape, not a method.</p></div>

<h2 id="s2">2. Related systems</h2>
<p>Four lines of work touch this problem. <b>Agent-aware serving</b> (ThunderAgent-style pause and resume, Continuum-style KV time-to-live from tool start) manages the working set reactively: when memory is tight, pause or drop the session whose KV is least likely to be needed soon, judged from history or a fixed TTL. These are our baselines B2 and the working-set arm. <b>Demand-aware scheduling</b> (ConServe-style co-serving of latency-sensitive and best-effort work) exploits the same two-class split we use, one layer down. <b>Serving frameworks</b> (NVIDIA Dynamo with its KV router, planner and <code>nvext.agent_hints</code>; vLLM and SGLang priority scheduling) provide the actuation points: a priority field the engine honours and metrics the proxy can scrape. <b>Sandbox platforms</b> (DSec, above) provide the execution side and its measured statistics. What none of them do is forecast the fleet's demand from the sessions in flight and shape admission against that forecast. That is the gap ATFM occupies, and it is why the evaluation compares against both a reactive working-set baseline and an oracle.</p>

<h2 id="s3">3. Design</h2>
<p>The design is recorded as twelve decisions (D1 to D12) in the specification, each with the alternative rejected and the reason. The ones that shape everything below: the class-aware queue lives in a proxy above Dynamo with a <i>global</i> admission window and a stable priority <i>tier</i> written into the request hints (D1); policy claims come only from closed-loop runs, never from replayed traces (D12); forecasts are Monte Carlo samples per horizon, not moments (D5); every predictor implements one interface so that gains are attributable to information used rather than code (D6); every controller fails open (D10); and KV tiering is future work because no shipped API pins or demotes one paused session's KV today (D3).</p>

<h3>3.1 System context</h3>
<figure><div class="plate"><img src="{f('fig1-context')}" alt="System context diagram"></div>
<figcaption><b>Figure 1. System context.</b> Two paths leave every harness: the request path (LLM calls through the proxy into Dynamo) and the state path (tool progress from the sidecar). Three kinds of people depend on the system: developers driving interactive agents, owners of background jobs with deadlines, and the platform operator who sets the trade-offs. Public trace corpora feed the offline evaluation.</figcaption></figure>

<h3>3.2 Containers</h3>
<figure><div class="plate"><img src="{f('fig2-container')}" alt="Container diagram"></div>
<figcaption><b>Figure 2. Containers (v1.1).</b> Row one is the request path. Row two is the event and forecast cycle: sidecar and proxy publish to the bus, the demand board consumes it and forecasts every 5 s, the controllers turn a snapshot into hold directives that return to the proxy. Nothing on the state path can block the request path. In development the JSONL bus <i>is</i> the trace source: the same events become the canonical trace table.</figcaption></figure>

<h3>3.3 Data model</h3>
<p>Everything, real or simulated, is expressed in one event schema and one flattened trace record (D4). Events are pydantic models discriminated by <code>kind</code>: <code>session.start/end</code>, <code>llm.request/first_token/done</code>, <code>tool.start/progress/data/end</code>, <code>spawn.request</code>, <code>worker.metrics</code>. The trace record has one row per LLM call together with the tool phase that call launched: timestamps for request, first token, last token, tool start and tool end; input and output lengths; tool name, backend and command signature; the progress events (<code>completed/total</code> with phase) and data events (named metrics such as an early-stop probability) observed during the tool; and parent session for fan-out. Adapters exist for TraceLab, AgentX and the sidecar's own events. A forecast snapshot holds, per target (KV blocks, prefill tokens), per class and per horizon, an array of Monte Carlo samples plus the fraction of demand that is endogenous (from sessions already in flight).</p>

<h3>3.4 What one turn moves through</h3>
<figure><div class="plate"><img src="{f('fig3-lifecycle')}" alt="Session turn lifecycle"></div>
<figcaption><b>Figure 3. Session turn lifecycle and what the board conditions on in each phase.</b> <code>tool_running</code> is where the information is: elapsed time, progress events and the backend's shared slowdown factor. <code>llm_pending</code> (tool ended, next request not yet sent) is modelled as a conditional gap, because the tool-end to next-request delay has a heavy tail in real traces (TraceLab p99 45 s, maximum 128 h). A session that never calls again is a no-return mass per tool.</figcaption></figure>

<h3>3.5 The queue, the index and the tiers</h3>
<p>Three queues exist: the proxy's hold queue (the only place a call can be delayed on purpose), the engine's own scheduler queue, and the router's choice among workers. The proxy keeps a global window of in-flight calls; when the window is full, the next call released is the one with the highest (tier, index). The tier is stable and coarse: 2 for interactive calls under deadline slack, 1 for interactive, 0 for background. It is written into <code>nvext.agent_hints</code> so that the router and engine keep class order for everything released. The index within a tier is a weighted score over expected service time, expected next-tool duration (a session that will vanish into a long tool is a good one to serve first) and context size, with a weight <i>beta</i> the operator sets. Because the window equals the batch size in the experiments so far, all queueing happens at the proxy (queue share 1.0 for proxy arms, 0 for native), which is intended: the proxy is the only place holds can be applied.</p>

<h3>3.6 Demand board and the predictor ladder</h3>
<figure><div class="plate"><img src="{f('fig4-ladder')}" alt="Predictor ladder"></div>
<figcaption><b>Figure 4. The predictor ladder.</b> Each rung adds one source of information. B0 and B1 are series predictors on the realized demand history alone. B2 uses per-tool duration history but ignores elapsed time (a Continuum-style time-to-live from tool start). M1 conditions the same distribution on elapsed time (survival). M2 adds a Bayesian rate filter over live progress events and a learned per-tool progress curve mapping reported progress to time fraction (cmake percent markers and test counts are rarely linear in time). M3 adds a shared per-backend slowdown factor. Every rung returns a distribution over time-to-next-call and next-call size from the same session-state input.</figcaption></figure>
<p>The fleet forecaster draws, for each session and each sample, a return time and a next-call size, adds one level of fan-out (children a tool may spawn) and an exogenous non-homogeneous Poisson arrival process for sessions not yet started, and sums into demand per horizon. Demand truth counts each session once per horizon: the KV required on its first resumption, not every hop. Dispersion calibration fits a per-class, per-horizon variance inflation out of sample within the training data on an overlaid holdout fleet.</p>

<h3>3.7 Ground delay: holding at a stated cost</h3>
<figure><div class="plate"><img src="{f('fig6-hold')}" alt="Hold semantics"></div>
<figcaption><b>Figure 5. Hold semantics and cost accounting.</b> The GDP-lite rule (from air-traffic ground delay programs) holds a deferrable call while the 90th percentile of forecast interactive demand over the next slot exceeds free KV capacity or free batch slots, where free capacity is what running requests do not hold, not merely what the LRU has not filled, and the slot test compares expected busy slots (calls × E[service] / slot) with free slots. A hold is capped in the simulator core at <code>max_hold_s</code> after arrival and re-consulted at expiry, so it lasts only while the constraint binds. Cost columns are measured, never assumed: seconds held, held session's resident blocks × hold time until eviction, displacement evictions attributed to held sessions, and prefix recomputed on resumption.</figcaption></figure>

<h3>3.8 Sidecar</h3>
<p>The sidecar wraps every tool subprocess in the harness. The result path is untouched and byte-exact (verified with invalid UTF-8 and a killed timeout); a separate state path parses the tool's output for progress (pytest counters, cmake and make percent markers, generic <code>k/N</code> counters, line rate) and publishes events off the request path. It classifies the command into a coarse tool class and, since this build, records a <i>command signature</i> (sizes normalised, attached flag digits kept, so <code>--rows 600</code> and <code>--rows 900</code> share a curve while <code>-j1</code> and <code>-j2</code> do not), which keys progress curves finer than the tool name. A launch gate lets a deferrable tool or spawn ask whether it may start. The sidecar is fail-open: a parser exception drops the line, a bus failure is ignored.</p>

<h3>3.9 Simulator</h3>
<figure><div class="plate"><img src="{f('fig5-sim')}" alt="Simulator structure"></div>
<figcaption><b>Figure 6. The closed-loop simulator.</b> Programs (synthetic from a workload spec, or built from a trace table) are sequences of turns with prompt growth, tool duration, progress reports and fan-out. A heap event loop drives a proxy queue and KV-aware workers (LRU with prefix reuse, batch limit, head-of-line priority). Every arm is one policy object exposing window, tier and index, an arrival-time hold decision and a per-tick update. Every call leaves a lifecycle row with where it waited and what any hold cost; serving metrics are paired across arms by seed with a bootstrap over sessions.</figcaption></figure>

<h2 id="s4">4. Implementation</h2>
<p>Python 3.12, uv, pydantic v2, numpy and pandas, FastAPI for the proxy and board, pytest; no Rust (D7). 55 source modules, about 4,600 lines, 151 tests (2 gated on a running Dynamo). The table lists what exists and its verification state.</p>
<div class="tblwrap"><table>
<caption><b>Table 1. Modules and their status.</b></caption>
<thead><tr><th>Area</th><th>Modules</th><th>What it does</th><th>Status</th></tr></thead><tbody>
<tr><td>Schema</td><td><code>schema/trace, forecast, events</code></td><td>Trace row and table (nested columns as JSON in parquet), forecast snapshot, event models</td><td><span class="status done">done</span></td></tr>
<tr><td>Traces</td><td><code>traces/tracelab, agentx, sidecar, transform, synthetic</code></td><td>Corpus adapters, family-aware split and Poisson overlay, event-to-trace adapter, synthetic fleet generator</td><td><span class="status done">done</span></td></tr>
<tr><td>Board</td><td><code>board/state, live, replay, forecaster, calibrate, predictors/*</code></td><td>Session registry, replayer, Monte Carlo fleet forecaster with exogenous arrivals, dispersion calibration, ladder B0 to M3</td><td><span class="status done">done</span></td></tr>
<tr><td>Evaluation</td><td><code>eval/forecast, coverage, resumption, serving</code></td><td>CRPS, pinball, coverage, surge lead time; signal coverage; per-session remaining-time scoring; serving metrics and paired bootstrap</td><td><span class="status done">done</span></td></tr>
<tr><td>Sidecar</td><td><code>sidecar/core, parsers, gate, minisweagent</code></td><td>Byte-exact tool wrapper, parser chain, launch gate, mini-SWE-agent local and Docker environments</td><td><span class="status done">done</span></td></tr>
<tr><td>Proxy</td><td><code>proxy/config, index, queue, app</code></td><td>OpenAI-compatible endpoint, global window, tiers and index, hold directives with cap, call log, gate</td><td><span class="status done">done</span></td></tr>
<tr><td>Dynamo</td><td><code>dynamo/local</code></td><td>Mocker workers plus frontend with file discovery; starts in about 4 s on a laptop</td><td><span class="status done">verified</span></td></tr>
<tr><td>Collection</td><td><code>collect/jobs, driver</code></td><td>Scripted Docker jobs (pytest suites, cmake builds, pipelines) through the sidecar</td><td><span class="status done">done</span></td></tr>
<tr><td>Simulator</td><td><code>sim/programs, engine, core, policies, forecast_arm</code></td><td>Programs, KV engine, event loop, six arms + ablation + same-rule oracle</td><td><span class="status done">reviewed, fixed</span></td></tr>
<tr><td>Experiments</td><td><code>experiments/h1, h2sim</code>; <code>scripts/*</code></td><td>H1 runner and sweeps, H2 regime runner, H1b leave-one-family-out scorer</td><td><span class="status done">done</span></td></tr>
<tr><td>Controllers</td><td>pre-staging, replica floor</td><td>Logged only in v1 (D3)</td><td><span class="status next">future</span></td></tr>
</tbody></table></div>
<p>The proxy was verified end to end against Dynamo v1.5 Mocker workers on this machine: three concurrent scripted sessions kept the window at two and gave the deadline-bound interactive session the top tier. The simulator was built test-first from a written plan, then reviewed by a fresh reviewer, which found three critical and seven important defects (a hold rule that measured LRU emptiness instead of capacity; a working-set arm whose working set included the paused sessions' own KV; hold cost conflated with ordinary window queueing; one-shot holds so the cap never engaged; ended sessions kept in the registry; throughput computed from the specified duration rather than the realized makespan; head-of-line skipping under priority; shared child objects across overlay copies). All were fixed with pinned tests before any result below was produced.</p>

<h2 id="s5">5. Evaluation method</h2>
<h3>5.1 Corpora and collections</h3>
<div class="tblwrap"><table>
<caption><b>Table 2. Data used.</b></caption>
<thead><tr><th>Source</th><th>Content</th><th>Size</th><th>Used for</th></tr></thead><tbody>
<tr><td>TraceLab</td><td>Real Claude Code and Codex sessions with per-round timestamps and context sizes</td><td>4,265 sessions, 357k rounds</td><td>H1a; weekly time-block split, held-out weeks overlaid as a Poisson fleet</td></tr>
<tr><td>AgentX (SemiAnalysis)</td><td>Claude Code sessions with real sub-agent groups; timestamps relative to trace start; inter-request gap is unlabeled tool-or-human time</td><td>393 sessions, 1,697 sub-agent groups</td><td>H1a second corpus; family split (root plus children) and family-aware overlay</td></tr>
<tr><td>Sidecar collections</td><td>Real tools in Docker: cmake builds of fmt at -j1 and -j2, data pipelines with linear or staged progress at four sizes, numpy's full test suite (49,696 tests)</td><td>18 + 2 sessions, 18 long phases, 51k progress events per numpy run</td><td>H1b per-session scoring</td></tr>
<tr><td>Synthetic programs</td><td>Workload specs: arrival rates, turn counts, prompt growth, lognormal tool durations with signal class, fan-out, think times</td><td>3,600 s per seed, 3 seeds per regime</td><td>H2 closed loop</td></tr>
</tbody></table></div>
<h3>5.2 Scoring</h3>
<p>Forecasts are scored per tick against realized demand with proper scoring rules: CRPS from samples, pinball loss at q90 and q95 (under-forecasts weighted 9x at q90, since a surge missed costs more than a surge over-called), 80 and 90% interval coverage, and surge lead time against the capacity line. History baselines see only realized windows, never the future. Per-session resumption scoring (H1b) places an observation every 15 s inside every tool phase longer than 30 s and scores each predictor's distribution of the remaining time against the truth, leaving one job family out at a time. Serving metrics (H2) are TTFT after a tool returns, session-weighted SLO attainment at a 2 s bound, background job completion time, deadline hit rate, recomputed prefill tokens, and the hold cost columns of Figure 5; arms are paired by seed and differences carry 95% bootstrap intervals over sessions.</p>

<h2 id="s6">6. Results</h2>
<h3>6.1 H1a: in-flight state versus history, two corpora</h3>
<div class="tblwrap"><table>
<caption><b>Table 3. TraceLab, seed 0, 200 sessions/h.</b> Mean q90 pinball loss of forecast KV blocks demanded within the horizon (lower is better). Predictors fit on held-in weeks; held-out weeks replayed as a fleet for 4 h.</caption>
<thead><tr><th>model</th><th class="n">10 s</th><th class="n">30 s</th><th class="n">2 min</th><th class="n">5 min</th><th class="n">15 min</th></tr></thead><tbody>
<tr><td>B0 constant</td><td class="n">10409</td><td class="n">10845</td><td class="n">13861</td><td class="n">16485</td><td class="n">43105</td></tr>
<tr><td>B1 Kalman</td><td class="n"><b>4588</b></td><td class="n">6615</td><td class="n">10597</td><td class="n">14186</td><td class="n">26101</td></tr>
<tr><td>B2 per-tool history, elapsed ignored</td><td class="n">40651</td><td class="n">38324</td><td class="n">33861</td><td class="n">30628</td><td class="n">26183</td></tr>
<tr><td>M1 survival on elapsed time</td><td class="n">5567</td><td class="n"><b>5254</b></td><td class="n"><b>5679</b></td><td class="n"><b>6604</b></td><td class="n"><b>7700</b></td></tr>
<tr><td>M2 progress (no progress events in this corpus)</td><td class="n">5538</td><td class="n">5251</td><td class="n">5654</td><td class="n">6611</td><td class="n">7671</td></tr>
</tbody></table></div>
<figure class="chart"><div class="two"><div>{chart_ratio(TRACELAB, "TraceLab: held-out weeks")}</div><div>{chart_ratio(AGENTX, "AgentX: held-out families")}</div></div>
<figcaption><b>Figure 7. H1a sweep, three seeds × three arrival rates.</b> Ratio of the Kalman baseline's q90 pinball loss to the elapsed-time model's (above 1 favours in-flight state). The advantage grows with horizon and fleet density on both corpora and is never below 1. AgentX 15-minute cells carry 95% CI half-widths of 50 to 60% of the mean (a few hundred held-out sessions dominated by long sub-agent sessions) and should be read as "about 3x"; all other cells are within 10 to 25%.</figcaption></figure>
<p>Demand at these horizons is 81 to 99% endogenous: it comes from sessions already in flight. The session models are under-dispersed (40 to 73% coverage of a 90% interval on TraceLab); out-of-sample calibration lifts that to 65 to 86% but not to 90, which says the remaining miss is bias on sessions and tools unseen in training rather than variance. B2 is the worst predictor on both corpora by a wide margin: ignoring elapsed time inside a heavy-tailed tool is the wrong thing to do, which is the quantitative case against fixed TTLs from tool start.</p>

<h3>6.2 H1b: live progress inside real long tools</h3>
<figure class="chart"><div>{chart_h1b()}</div>
<figcaption><b>Figure 8. H1b, leave-one-job-family-out, 202 observation points per model.</b> Mean q90 pinball loss on remaining seconds, per job family, for the three session predictors. With a linear signal (pipelines) M2 is near exact; with stage markers only it halves M1's error; on cmake builds it depends on whether a curve for that build's shape is in training. The numpy suite is the S-shaped case: 12% of tests done at a quarter of the wall time, 81% at three quarters. With the suite never seen, linear extrapolation buys nothing (14.6 vs 14.0); with one prior run in training, CRPS drops 2.9x against M1 (18.8 vs 53.7).</figcaption></figure>
<p>CRPS over all phases: B2 67.3, M1 59.0, M2 30.1. B2's small numbers on numpy and on the -j2 build are artifacts of near-identical durations across repeats and of a pooled draw that happens to sit above the truth; the TraceLab and AgentX fleets show what B2 does when durations vary. The gate for H1b was set in advance as "M2 beats M1 on a 45-minute concurrency-1 long-tool collection", and it passed on the varied-duration collection with q90 pinball 10.0 vs 28.1 before numpy was added.</p>

<h3>6.3 H2 in the simulator: six arms, three regimes</h3>
<div class="callout"><p><b>Read this section as ordering only.</b> The workload is synthetic and the engine's KV model (block LRU with prefix reuse, fixed prefill and decode rates) is unvalidated against a real engine. No absolute number here transfers to an H100. The simulator exists to say where each arm queues work and what a hold costs, so the GPU study measures the right things.</p></div>
<p>One engine of 8,000 KV blocks and batch 8 (6,000 and 6 in the loaded regime), Dynamo-style priority honoured with head-of-line order kept. Interactive sessions: 6 turns, 20 s think time, bash tools of a few seconds. Background: 10 turns, deadline 1,800 s, tools bash 3 s / pytest 180 s with a strong signal / build 400 s with a weak signal (<i>long_tool</i>), or bash 2 s / pytest 20 s (<i>short_tool</i>). Arms: <b>native</b> (engine priority only), <b>proxy_rules</b> (window, tiers, index, no holds), <b>forecast_M1</b> and <b>forecast_M2</b> (rules plus the GDP-lite hold on the board's forecast), <b>forecast_M2_nohold</b> (forecast only in the index), <b>oracle</b> (reads the event heap: true arrivals in the next slot, occupancy rule) and <b>working_set</b> (ThunderAgent-style: pause background programs and offload their KV when the working set exceeds budget; hysteresis release). Hold cap 600 s in these runs.</p>
<div class="tblwrap"><table>
<caption><b>Table 4. Loaded long-tool regime (360 interactive and 600 background sessions/h), 3 seeds, paired against native.</b> 95% bootstrap intervals in brackets.</caption>
<thead><tr><th>arm</th><th class="n">SLO sessions</th><th class="n">diff vs native</th><th class="n">TTFT after tool p95 (s)</th><th class="n">background JCT diff (s)</th><th class="n">deadline hit</th><th class="n">recomputed prefill tokens</th><th class="n">held KV block-s</th><th class="n">holds at cap</th></tr></thead><tbody>
<tr><td>native</td><td class="n">0.851</td><td class="n">—</td><td class="n">2.68</td><td class="n">—</td><td class="n">0.190</td><td class="n">4.79e7</td><td class="n">0</td><td class="n">0</td></tr>
<tr><td>proxy_rules</td><td class="n">0.867</td><td class="n">+0.014 [−0.002, +0.028]</td><td class="n">3.01</td><td class="n">−35 [−90, +17]</td><td class="n">0.247</td><td class="n">3.82e7</td><td class="n">0</td><td class="n">0</td></tr>
<tr><td>forecast_M1</td><td class="n">0.941</td><td class="n"><b>+0.088 [+0.074, +0.101]</b></td><td class="n">2.16</td><td class="n">+790 [736, 842]</td><td class="n">0.119</td><td class="n">4.33e7</td><td class="n">4.2e6</td><td class="n">1218</td></tr>
<tr><td>forecast_M2</td><td class="n">0.946</td><td class="n"><b>+0.093 [+0.080, +0.105]</b></td><td class="n">2.12</td><td class="n">+789 [739, 837]</td><td class="n">0.122</td><td class="n">4.31e7</td><td class="n">4.2e6</td><td class="n">1193</td></tr>
<tr><td>forecast_M2_nohold</td><td class="n">0.841</td><td class="n">−0.013 [−0.028, +0.001]</td><td class="n">3.23</td><td class="n">−35 [−97, +10]</td><td class="n">0.243</td><td class="n">4.14e7</td><td class="n">0</td><td class="n">0</td></tr>
<tr><td>oracle (heap)</td><td class="n">0.923</td><td class="n">+0.070 [+0.054, +0.083]</td><td class="n">2.51</td><td class="n">+1141 [1090, 1190]</td><td class="n">0.091</td><td class="n">4.10e7</td><td class="n">2.2e6</td><td class="n">805</td></tr>
<tr><td>working_set</td><td class="n">0.929</td><td class="n">+0.076 [+0.063, +0.089]</td><td class="n">2.29</td><td class="n">+2082 [2035, 2136]</td><td class="n">0.114</td><td class="n">4.66e7</td><td class="n">0</td><td class="n">1096</td></tr>
</tbody></table></div>
<figure class="chart"><div>{chart_h2()}</div>
<figcaption><b>Figure 9. The trade in the loaded regime.</b> Left: interactive SLO gain over native. Right: what background work pays for it. The forecast arms and the working-set baseline reach similar SLO; the working set costs 2.6x as much background time and 58% more recomputed prefill in the long-tool regime, because pause and resume cycles evict and recompute whole contexts. Rules alone and the no-hold ablation are within noise of native.</figcaption></figure>
<p>Three readings are safe. First, holds are what buy the interactive tail: the ablation identical to forecast_M2 except that it never holds is indistinguishable from plain rules. Second, holds are not free: 24% more background completion time, 7 points fewer deadlines, 4.2 million held KV block-seconds and about 1,200 holds per seed that ran to the 600 s cap; held calls then still wait at the window (median proxy queue 579 s for held calls, 1 s for unheld background). Third, demand-aware holds reach the same SLO as occupancy-triggered pausing at 40% of its background cost. In the two base regimes (120 and 240 sessions/h) every arm sits at 0.99 to 1.00 SLO and only the p95 TTFT after a tool moves (22% lower with holds in long_tool, 8x lower in short_tool), so those regimes cannot rank arms on the SLO.</p>
<p>Two things the simulator did <i>not</i> show. M1 and M2 do not separate: their TTFT, completion time and hold costs are within noise, because interactive sessions in these programs run only short tools, so progress prediction never enters the hold rule. And the heap-reading oracle is not an upper bound on the forecast arms: it uses a different rule (occupancy of the next slot from true arrivals), holds less and gets a worse tail. Both gaps are addressed by runs in flight (Section 8).</p>

<h2 id="s7">7. Limitations</h2>
<ul>
<li><b>No GPU results.</b> Every policy number is from the simulator; the KV model is the least validated component. The Dynamo Mocker path exercises the request plumbing, not memory.</li>
<li><b>Eight job families, one machine, no humans.</b> H1b covers tools with parseable output; it says nothing about tools that report nothing until they finish or about human think time.</li>
<li><b>Curves keyed by tool name mislead when shapes mix.</b> Adding -j1 builds to training moved the -j2 error from 3.8 to 15.6. Command signatures (this build) address it but have not yet been re-scored on the collections.</li>
<li><b>Under-dispersion is bias, not variance.</b> Calibration cannot reach 90% coverage at short horizons; unseen sessions and tools need a fallback prior, not a wider one.</li>
<li><b>Fan-out is one level deep</b> in the forecaster and the simulator.</li>
<li><b>Hold cap of 600 s</b> in the reported runs is the core default, not the proxy's 60 s.</li>
</ul>

<h2 id="s8">8. What is next</h2>
<figure><div class="plate"><img src="{f('fig7-roadmap')}" alt="Demonstration ladder"></div>
<figcaption><b>Figure 10. Demonstration ladder.</b> Green is built and evidenced; grey is next. Explicit KV placement waits for a demonstrated mechanism (D3) and for the H100 study to show that holds pay.</figcaption></figure>
<h3>8.1 Runs in flight at the time of writing</h3>
<ul>
<li><b>interactive_long_tool regime</b> (new): interactive sessions also run 120 s test suites with a strong signal and 300 s builds with a weak one, at loaded rates, hold cap 60 s. This is where M1 and M2 can separate in closed loop, because <i>when</i> interactive sessions return is now what the hold rule has to predict.</li>
<li><b>oracle_rule arm</b> (new): the same GDP-lite rule and index fed the true first-call demand per horizon read from the event heap. Whatever the forecast arms lose against it is forecast error; whatever it loses against native on background cost is the rule itself.</li>
<li><b>Loaded regime at a 60 s cap</b>, to see how much of the background cost was the long cap.</li>
</ul>
<h3>8.2 The two-GPU study (L2)</h3>
<p>Two H100s running Dynamo with vLLM workers, the proxy in front, the same six arms plus the ablation and the same-rule oracle, mini-SWE-agent sessions on scripted Docker jobs as the background class and replayed interactive programs as the foreground. What must be recorded on the real engine is exactly the columns the simulator now logs: hold seconds, held KV block-seconds, evictions attributed to holds, recomputed prefill after resumption, holds at cap, and TTFT after tool return. The regimes start from the loaded rates, not the base rates, because the base rates never stress a 2 s SLO. The study answers whether the simulator's ordering survives contact with a real KV cache; the design of the memory tiers waits on that answer.</p>
<h3>8.3 Model work</h3>
<ul>
<li>Re-score H1b with signature-keyed curves and an online per-session curve adaptation for suites seen once.</li>
<li>A fallback prior for unseen tools, which is where the remaining coverage miss lives.</li>
<li>Replay real trace tables (TraceLab, AgentX) through the simulator arms with prompt sizes from the corpus, as the calibration step before L2.</li>
</ul>

<h2 id="refs">References</h2>
<ol class="refs">
<li>Project specification v1.1: <code>docs/superpowers/specs/2026-09-22-atfm-architecture-design.md</code>; master plan <code>detail.md</code>.</li>
<li>Results notes: <code>docs/research/2026-09-22-h1-first-results.md</code>, <code>2026-09-23-h1-second-corpus-and-h1b.md</code>, <code>2026-09-23-l1-results.md</code>, <code>2026-09-24-h2sim-first-results.md</code>, <code>2026-09-27-dsec-paper-notes.md</code>.</li>
<li>DeepSeek, "DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale", arXiv 2609.22978, September 2026.</li>
<li>NVIDIA Dynamo v1.5: frontend, KV router, planner, Mocker workers; <code>nvext.agent_hints</code> request extension.</li>
<li>TraceLab agent session corpus; AgentX (SemiAnalysis) Claude Code session corpus.</li>
<li>ThunderAgent (agent-aware pause and resume of sessions), Continuum (KV time-to-live from tool start), ConServe (co-serving latency-sensitive and best-effort LLM work): the baselines this program compares against.</li>
</ol>
<p class="foot">Diagrams authored in reladraw from the architecture sources; charts drawn to scale from the run outputs under <code>runs/</code>. Build script: <code>docs/paper/build_paper.py</code>.</p>
</div>
"""


if __name__ == "__main__":
    out = HERE / "atfm-paper.html"
    out.write_text(html())
    print(out, len(out.read_bytes()))
