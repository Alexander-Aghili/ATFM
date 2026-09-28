# ATFM research paper

For code setup and current feature status, see the [project README](../../README.md)
and [implementation status](../status.md). This guide covers publication sources
and frozen evidence; rebuilding the paper does not validate the serving system.

Open [atfm-paper.pdf](atfm-paper.pdf) for the paginated paper. The HTML files preserve an earlier revision and do not include all LaTeX additions.

## Edit and rebuild (LaTeX)

The primary editable manuscript is now `atfm-paper.tex`. It uses native LaTeX equations, tables, section numbering, cross-references, and BibTeX citations. `references.bib` contains the bibliography; `latex-fig/` contains standalone vector PDF figures and their SVG sources. `atfm-paper.pdf` is compiled from this LaTeX source.

From the repository root:

```bash
python3 docs/paper/build_latex.py
# Also refresh the portable source bundle:
python3 docs/paper/build_latex.py --bundle
```

Or compile directly from this directory:

```bash
latexmk -pdf atfm-paper.tex
```

A standard TeX Live installation with `latexmk`, pdfLaTeX, BibTeX, `newtx`, `microtype`, `tabularx`, `booktabs`, `makecell`, `caption`, `fvextra`, `placeins`, `needspace`, `natbib`, `xurl`, and `hyperref` is sufficient. No Python, raw traces, browser, or Inkscape is needed for the direct LaTeX build. The Python wrapper places auxiliary files in `docs/paper/tmp/latex/`, checks for unresolved references and overflowing boxes, then publishes the PDF.

`atfm-latex.zip` is a portable source bundle suitable for Overleaf: select `atfm-paper.tex` as the main document and pdfLaTeX as the compiler. It includes the bibliography, vector figures, and frozen result summaries.

To refresh the figures from the existing result snapshots (requires Python with NumPy, pandas and Matplotlib, Inkscape, and Node/npx with reladraw):

```bash
python3 docs/paper/build_latex.py --figures
```

The manuscript and native table cells are editable directly in `atfm-paper.tex`; figure refresh does not overwrite them. If experiment results change, update the tables and associated claims deliberately, then rebuild. `results/manifest.json` identifies the frozen CSV sources and hashes. Original raw inputs and run outputs remain in the repository's ignored `data/` and `runs/` directories.

## Earlier HTML version

`manuscript.html`, `paper.css`, and `build_paper.py` preserve the earlier web publication. They are not synchronized automatically with edits to the LaTeX manuscript. Run `python3 docs/paper/build_paper.py` to rebuild its HTML files. Its optional `--pdf` output is now `atfm-paper-html.pdf`, so it cannot overwrite the primary LaTeX PDF.

The architecture source is `fig/fig-system.reladraw`. To edit and propagate that figure:

```bash
npx reladraw docs/paper/fig/fig-system.reladraw -o docs/paper/fig/fig-system.svg
python3 docs/paper/build_latex.py --figures
```

Earlier architecture and roadmap diagrams remain in `fig/` as historical assets but are not embedded in the revised manuscript.

## Evidence and revision notes

The revision draws on the architecture specification, master plan, research notes, experiment configurations, implemented models and metrics, and completed local run summaries. It preserves measured results while correcting the following interpretations:

- Sweep figures use ratios of mean losses across seeds; individual seeds can favor the Kalman baseline at short horizons.
- B0 is a last-value persistence predictor. B2 subtracts elapsed time without survival conditioning; it is not a reproduction of Continuum.
- The expanded H1b score file has nine scored job families and 202 observations per model. Its M1/M2 loss ratios are 2.80 for q90 pinball and 1.96 for CRPS. Repeated observations are not independent replicates.
- Session-weighted SLO is the average within-session fraction of eligible calls meeting the target. The loaded native point estimate is 0.853 from `paired.csv`; the older draft mixed in a different aggregation.
- H2 intervals are a 300-draw paired bootstrap over common session IDs pooled across three seeds, not a seed-level uncertainty analysis.
- The completed long-interactive experiment is included, including its lack of an M2 policy advantage. The 60-second-cap loaded run completed after the first cutoff and is included as Section 6.5 (a controlled cap ablation); the GDP-lite v2 reruns of both regimes are included as Section 6.6; the KV placement runs, including the corrected true-return-time reruns, are Section 6.7; the third batch (direct contrasts, size and class variants, keep-alive touch arms) is Section 6.8; the ordering-artefact correction, corrected true-return-time reruns and queue-aware placement are Section 6.9; the index diagnostic is reported in Section 6.6.
- Working-set and lookahead policies are local diagnostic baselines, not full published systems or optimal upper bounds.
- First-resumption KV demand is not instantaneous resident occupancy, and prompt demand is not cache-miss-adjusted prefill work.
- Current signature-aware progress support is separated from historical collection scores. Hardware validation, a deployed Redis topology, real-worker KV placement, and external replica scaling are not presented as completed results; simulated placement experiments are reported separately.

External references were checked against primary paper, dataset, and vendor pages. The paper supplies numbered linked references and an appendix identifying local evidence.

## Verification

Rebuilding does not rerun experiments. Inspect every exported PDF page after layout changes, and check that tables and plot labels remain legible. The retained CSVs allow numeric elements to be rebuilt without raw traces; they do not substitute for an experiment release with input versions and historical code commits.

## LaTeX conversion checks

The original conversion preserved the five-regime manuscript, including the controlled 60-second cap ablation. Subsequent results and figures are recorded in the current LaTeX source and visual inventory. Two consistency corrections were made: the appendix now identifies that completed run, and Section 6.5 no longer describes the working-set row as unchanged when its table values change. The source compiles without missing citations, unresolved references, missing glyphs, or overfull boxes. All rendered pages were inspected.

## Expanded visual and mathematical edition

The LaTeX edition adds a lifecycle timeline, two attributed original prior-paper diagrams, a forecast-flow diagram, worked survival/progress illustrations, calibration plots, cross-regime policy effects, a controlled-cap comparison, seed-level resource costs, and a technical research roadmap. `VISUALS.md` records their sources and interpretation. `build_visuals.py` regenerates scientific charts; toy examples are explicitly labeled illustrative. Prior-work originals and reuse provenance are retained under `sources/prior-work/`.

Section 8 is a technical future-work plan. Read its scope alongside the results sections: simulated KV placement and implemented diagnostics are distinct from proposed occupancy control, real-worker tier placement, hardware validation, and external replica scaling. Equations define their components locally and distinguish implemented rules from proposed extensions.

The probability expansion adds return/no-return mixtures, pending-gap convolution, hazard conditioning, Gamma rate updates, dependence-aware fleet variance, compound-Poisson moments, tail-risk measures, Monte Carlo error, quantile-cost interpretation, and the exact clipped calibration map. The probability expansion adds six figures; consult `VISUALS.md` and the manuscript for the current figure inventory. Appendix C maps the mathematics to code and the research plan; illustrative derivations are explicitly separated from implemented algorithms and benchmark evidence. `probability_visuals.py` is included in the source bundle and invoked by `--figures`.

## HTML evidence synchronization

The LaTeX abstract now introduces the problem, approach, and qualitative findings without detailed experimental numbers. The results include the completed no-index diagnostic, both KV-eviction workloads, and corrected lookahead reruns from the updated HTML and frozen CSVs. The dated revision notes describe successive manuscript states. For the current set of included placement variants and direct contrasts, consult the LaTeX results sections and frozen CSV manifest; hardware validation remains separate. Native-referenced intervals are not described as equivalence tests or direct M1/M2 comparisons.

September 27 integration update: LaTeX includes Dynamo/LMCache background, code-level adapter boundaries, the deployment wiring check, and the latest placement/touch diagnostics. `fig/integration.reladraw` generates the integration diagram; `integration_visuals.py` plots frozen place/touch2 contrasts. Regenerate with `python3 build_latex.py --figures --bundle`. Backend actuation and GPU benefit remain unvalidated.

Timed Appendix D example: `python3 timed_cache_example.py` validates the synthetic schedules and regenerates `results/timed-cache-example.json`, `latex-fig/timed-cache-timeline.{pdf,svg}`, and `latex-fig/timed-cache-residency.{pdf,svg}`. Inputs are illustrative, including manual online forecasts; no real GPU or fitted ATFM performance is claimed. The optimal schedule attains the 12 ms total-flow lower bound.

## Control implementation revisions

`control-implementation.tex`, included by `atfm-paper.tex`, records incremental
JSONL ingestion, batched hold delivery, indexed admission, planner range pruning,
and empirical-sampling choices. It distinguishes component CPU measurements from
simulation outcomes and real-worker capacity. [Development decisions](../development/control-scaling.md)
contain contracts, limitations, test coverage, and benchmark commands. The source
bundle includes this file; regenerate with `python3 docs/paper/build_latex.py --bundle`.
