# GPU results and caching tradeoffs

[Read the 25-page PDF](atfm-gpu-results-2026-09-30.pdf) (30 September edition) or the
[editable report](report.md). Fourteen figures distinguish measurements, derived
capacity bounds, and illustrative timing/bandwidth scenarios.

The report covers the 28-29 September RTX/H100 compatibility checks and first
public replays, round 2 (first complete large root, CPU-tier pair), round 3
(seeded random repeats, Blackwell), stage C (retrieval paths) and stage E
(forecast-driven warming, inconclusive). It does **not** establish an ATFM
policy speedup or official BFCL accuracy. Plots read the committed evidence under
`docs/research/results/` through `evidence.py`; `measured_charts.py` draws the
round 2-3, stage C and stage E figures.

## Rebuild

From the repository root, with Python 3.12, `uv`, and DejaVu Sans fonts installed:

```bash
uv venv /tmp/atfm-results-pdf-env
uv pip install --python /tmp/atfm-results-pdf-env/bin/python \
  -r docs/research/gpu-results-report/requirements.txt
/tmp/atfm-results-pdf-env/bin/python docs/research/gpu-results-report/build.py
```

The builder writes the PDF here and under `output/pdf/`, and regenerates
`figures/`. It changes no runtime code or experiment evidence. `report.md`
contains the text and explicit page boundaries; `charts.py` creates the plots;
`build.py` handles typography and layout. These helpers retain the project's
20-line function limit.

Latency and BFCL plots read the committed JSON evidence directly. Counter and
occupancy values reproduce the dated report; capacity plots use the measured
147,456-byte/token layout. Hypothetical bandwidth and timing inputs are labeled
inside their figures and captions. The report links immutable experiment
revisions and retains failed attempts separately from accepted runs.

## Review

```bash
pdfinfo output/pdf/atfm-gpu-results-2026-09-30.pdf
mkdir -p tmp/pdfs
pdftoppm -scale-to 1400 -png \
  output/pdf/atfm-gpu-results-2026-09-30.pdf tmp/pdfs/results
```

Inspect all pages after editing, especially long tables and page breaks. The
current report has 25 pages, fourteen labeled figures, and links to raw records and
archive checksums. The 29 September edition is superseded; its source is in git
history at `739f902`.
