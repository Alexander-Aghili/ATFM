"""Build the paper from an editable manuscript and frozen, traceable result CSVs.

    python3 docs/paper/build_paper.py
    python3 docs/paper/build_paper.py --pdf

The HTML is self-contained and uses local/system fonts. --pdf requires Chrome.
"""
from __future__ import annotations

import argparse
import base64
import csv
import html as html_lib
import math
from pathlib import Path
import shutil
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
DATA = HERE / 'results'
FIG = HERE / 'fig'
HORIZONS = [10, 30, 120, 300, 900]


def read_csv(name):
    with (DATA / name).open(newline='') as file:
        return list(csv.DictReader(file))


def ratios(corpus):
    rows = read_csv(f'sweep_{corpus}.csv')
    values = {(r['model'], int(float(r['rate'])), int(float(r['h']))): float(r['mean'])
              for r in rows if r['class'] == 'interactive'}
    return {rate: [values['B1_kalman', rate, h] / values['M1_survival', rate, h]
                   for h in HORIZONS] for rate in [100, 200, 400]}


TRACELAB, AGENTX = ratios('tracelab'), ratios('agentx')
FAMILIES = [('build-fmt-j1', 'fmt build -j1'), ('build-fmt-j2', 'fmt build -j2'),
            *[(f'pipe-s-{v}', f'pipeline linear {v}') for v in 'abcd'],
            *[(f'pipe-w-{v}', f'pipeline staged {v}') for v in 'ab'],
            ('pytest-numpy', 'numpy (unseen)'), ('all phases', 'all phases')]
H1B_ROWS = read_csv('h1b_summary.csv')
H1B = [(label, *[float(next(r['pinball90'] for r in H1B_ROWS
                          if r['family'] == family and r['model'] == model))
                 for model in ['B2', 'M1', 'M2']]) for family, label in FAMILIES]
ARM_LABELS = {'native': 'Native', 'proxy_rules': 'Rules', 'forecast_M1': 'Forecast M1',
              'forecast_M2': 'Forecast M2', 'forecast_M2_nohold': 'M2, no holds',
              'oracle': 'Heap lookahead', 'oracle_rule': 'Same-rule lookahead',
              'working_set': 'Working set'}


def paired(run):
    return {(r['arm'], r['metric']): r for r in read_csv(f'{run}__paired.csv')}


LOADED = paired('h2sim_long_tool_loaded')
H2 = [(ARM_LABELS[a], *[float(LOADED[a, metric][field])
                       for metric in ['slo_attainment_sessions', 'bg_jct_mean']
                       for field in ['diff_mean', 'diff_ci_lo', 'diff_ci_hi']])
      for a in ARM_LABELS if a not in ['native', 'oracle_rule']]


def fig_uri(name):
    return 'data:image/svg+xml;base64,' + base64.b64encode((FIG / f'{name}.svg').read_bytes()).decode()

def chart_ratio(data: dict, title: str) -> str:
    W, H, L, R, T, B = 460, 300, 52, 34, 48, 44
    xs = [math.log10(h) for h in HORIZONS]
    x0, x1 = xs[0], xs[-1]
    ymax = 6.0
    def X(v): return L + (v - x0) / (x1 - x0) * (W - L - R)
    def Y(v): return T + (1 - v / ymax) * (H - T - B)
    out = [f'<svg class="ratio" viewBox="0 0 {W} {H}" role="img" aria-label="{title}: ratio of Kalman baseline loss to elapsed-time model loss versus horizon at three arrival rates">']
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
    # legend (fixed order, same for both panels)
    for i, rate in enumerate(sorted(data)):
        out.append(f'<line x1="{L+8+i*110}" x2="{L+30+i*110}" y1="{T-14}" y2="{T-14}" class="s{i+1} line"/>'
                   f'<text x="{L+36+i*110}" y="{T-10}" class="tick">{rate}/h</text>')
    out.append("</svg>")
    return "".join(out)


def chart_h1b() -> str:
    rows = H1B
    W, L, R, T = 720, 150, 16, 26
    rh, gap = 30, 10
    H = T + len(rows) * (rh + gap) + 34
    vmax = 76.0
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
            out.append(f'<text x="{X(v)+4:.1f}" y="{yy+bh-1:.1f}" class="val">{v:.1f}</text>')
    for i, n in enumerate(("B2 history", "M1 elapsed time", "M2 progress")):
        out.append(f'<rect x="{L+8+i*180}" y="{T-22}" width="12" height="10" rx="2" class="s{i+1} bar"/><text x="{L+24+i*180}" y="{T-13}" class="tick">{n}</text>')
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
            for bound in (lo, hi):
                o.append(f'<line x1="{X(bound):.1f}" x2="{X(bound):.1f}" y1="{y-3:.1f}" y2="{y+3:.1f}" class="ci"/>')
            o.append(f'<circle cx="{X(m):.1f}" cy="{y:.1f}" r="3" class="s1 dot"/>')
        o.append("</g>")
        return "".join(o), H
    a, H = panel(0, "Interactive SLO difference", [(n, m, lo, hi) for n, m, lo, hi, *_ in H2], -0.04, 0.12,
                 lambda g: f"{g:+.2f}", [-0.04, 0, 0.04, 0.08, 0.12])
    b, _ = panel(370, "Background JCT difference (s)", [(n, j, lo, hi) for n, _, _, _, j, lo, hi in H2], -200, 2200,
                 lambda g: f"{g:+d}", [0, 500, 1000, 1500, 2000])
    return (f'<svg viewBox="0 0 740 {H}" role="img" aria-label="H2 loaded regime: paired differences against the native arm with 95% bootstrap intervals, interactive SLO on the left and background completion time on the right">'
            + a + b + "</svg>")



def cell(value, numeric=False):
    return f'<td class="{"n" if numeric else ""}">{html_lib.escape(str(value))}</td>'


def h1_table():
    rows = read_csv('h1_tracelab_r200__metrics.csv')
    names = {'B0_constant': 'B0 persistence', 'B1_kalman': 'B1 Kalman', 'B2_history': 'B2 history',
             'M1_survival': 'M1 survival', 'M2_progress': 'M2 progress'}
    out = []
    for model, label in names.items():
        values = [next(float(r['pinball90']) for r in rows
                       if r['model'] == model and r['target'] == 'kv_blocks'
                       and r['class'] == 'interactive' and float(r['h']) == h) for h in HORIZONS]
        out.append('<tr>' + cell(label) + ''.join(cell(f'{v:,.0f}', True) for v in values) + '</tr>')
    return ''.join(out)


def policy_table(run):
    data = paired(run)
    out = []
    for arm, label in ARM_LABELS.items():
        if (arm, 'bg_jct_mean') not in data:
            continue
        slo, jct = data[arm, 'slo_attainment_sessions'], data[arm, 'bg_jct_mean']
        def delta(r, scale, precision):
            return (f"{float(r['diff_mean'])*scale:+.{precision}f} "
                    f"[{float(r['diff_ci_lo'])*scale:+.{precision}f}, "
                    f"{float(r['diff_ci_hi'])*scale:+.{precision}f}]")
        vals = [f"{100*float(slo['mean']):.1f}", '-' if arm == 'native' else delta(slo, 100, 1),
                f"{float(jct['mean']):,.0f}", '-' if arm == 'native' else delta(jct, 1, 0)]
        out.append('<tr>' + cell(label) + ''.join(cell(v, True) for v in vals) + '</tr>')
    return ''.join(out)


def cost_table():
    rows = read_csv('h2sim_long_tool_loaded__metrics.csv')
    out = []
    for arm, label in ARM_LABELS.items():
        selected = [r for r in rows if r['arm'] == arm]
        if not selected:
            continue
        def mean(key):
            return sum(float(r[key]) for r in selected) / len(selected)
        deadline = 100 * float(LOADED[arm, 'deadline_hit_rate']['mean'])
        vals = [f"{mean('ttft_after_tool_p95'):.2f}", f'{deadline:.1f}',
                f"{mean('recomputed_prefill_tokens')/1e6:.2f}",
                f"{mean('hold_kv_block_s')/1e6:.2f}", f"{mean('caps'):,.0f}"]
        out.append('<tr>' + cell(label) + ''.join(cell(v, True) for v in vals) + '</tr>')
    return ''.join(out)


def build():
    template = (HERE / 'manuscript.html').read_text()
    replacements = {'CSS': (HERE / 'paper.css').read_text(), 'SYSTEM': fig_uri('fig-system'),
                    'H1_TABLE': h1_table(), 'H1_CHART': '<div class="two">' +
                    chart_ratio(TRACELAB, '(a) TraceLab') + chart_ratio(AGENTX, '(b) AgentX') + '</div>',
                    'H1B_CHART': chart_h1b(), 'H2_CHART': chart_h2(),
                    'LOADED_TABLE': policy_table('h2sim_long_tool_loaded'),
                    'LONG_TABLE': policy_table('h2sim_interactive_long'),
                    'CAP60_TABLE': policy_table('h2sim_long_tool_loaded_cap60'), 'COST_TABLE': cost_table()}
    for key, value in replacements.items():
        template = template.replace('{{' + key + '}}', value)
    if '{{' in template:
        raise ValueError('Unexpanded manuscript placeholder')
    for name in ['atfm-paper.html', 'atfm-paper-print.html']:
        (HERE / name).write_text(template)
        print(HERE / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pdf', action='store_true')
    args = parser.parse_args()
    build()
    if args.pdf:
        chrome = shutil.which('google-chrome') or shutil.which('chromium')
        if not chrome:
            parser.error('--pdf requires google-chrome or chromium')
        with tempfile.TemporaryDirectory(prefix='atfm-paper-chrome-') as profile:
            subprocess.run([chrome, '--headless=new', '--disable-gpu', '--no-sandbox',
                            '--no-pdf-header-footer', f'--user-data-dir={profile}',
                            f'--print-to-pdf={HERE / "atfm-paper.pdf"}',
                            (HERE / 'atfm-paper-print.html').as_uri()], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        print(HERE / 'atfm-paper.pdf')


if __name__ == '__main__':
    main()
