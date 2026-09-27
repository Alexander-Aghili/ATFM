# ATFM research paper

Open `atfm-paper.pdf` for the paginated paper or `atfm-paper.html` for the self-contained web version.

## Edit and rebuild

- `manuscript.html`: manuscript source, citations, equations, and captions.
- `paper.css`: screen and A4 print typography.
- `build_paper.py`: charts, numeric tables, HTML generation, and optional PDF export.
- `fig/fig-system.reladraw`: the paper's architecture figure source.
- `results/`: frozen experiment summaries used by the paper. `manifest.json` records source paths, hashes, and capture commit. Original experiment inputs and outputs remain under the repository's ignored `data/` and `runs/` directories.

From the repository root:

```bash
python3 docs/paper/build_paper.py
python3 docs/paper/build_paper.py --pdf
```

The builder uses only the Python standard library. PDF export additionally requires `google-chrome` or `chromium`. It uses a temporary browser profile, local fonts, and no remote web assets. Both HTML output names intentionally contain the same document; print CSS supplies pagination and page numbers.

To edit the diagram:

```bash
npx reladraw docs/paper/fig/fig-system.reladraw -o docs/paper/fig/fig-system.svg
python3 docs/paper/build_paper.py --pdf
```

Earlier architecture and roadmap diagrams remain in `fig/` as historical assets but are not embedded in the revised manuscript.

## Evidence and revision notes

The revision draws on the architecture specification, master plan, research notes, experiment configurations, implemented models and metrics, and completed local run summaries. It preserves measured results while correcting the following interpretations:

- Sweep figures use ratios of mean losses across seeds; individual seeds can favor the Kalman baseline at short horizons.
- B0 is a last-value persistence predictor. B2 subtracts elapsed time without survival conditioning; it is not a reproduction of Continuum.
- The expanded H1b score file has nine scored job families and 202 observations per model. Its M1/M2 loss ratios are 2.80 for q90 pinball and 1.96 for CRPS. Repeated observations are not independent replicates.
- Session-weighted SLO is the average within-session fraction of eligible calls meeting the target. The loaded native point estimate is 0.853 from `paired.csv`; the older draft mixed in a different aggregation.
- H2 intervals are a 300-draw paired bootstrap over common session IDs pooled across three seeds, not a seed-level uncertainty analysis.
- The completed long-interactive experiment is included, including its lack of an M2 policy advantage. The separate 60-second-cap loaded run was incomplete at the evidence cutoff and is not used.
- Working-set and lookahead policies are local diagnostic baselines, not full published systems or optimal upper bounds.
- First-resumption KV demand is not instantaneous resident occupancy, and prompt demand is not cache-miss-adjusted prefill work.
- Current signature-aware progress support is separated from historical collection scores. Hardware validation, Redis deployment, KV placement, and replica control are not presented as completed results.

External references were checked against primary paper, dataset, and vendor pages. The paper supplies numbered linked references and an appendix identifying local evidence.

## Verification

Rebuilding does not rerun experiments. Inspect every exported PDF page after layout changes, and check that tables and plot labels remain legible. The retained CSVs allow numeric elements to be rebuilt without raw traces; they do not substitute for an experiment release with input versions and historical code commits.
