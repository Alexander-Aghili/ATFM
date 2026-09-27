# Visual evidence and reconstruction

The reading sequence is observe → predict → aggregate → act → evaluate → validate. All empirical figures use frozen CSVs; no experiment was rerun for this editorial revision. Conceptual examples do not imply measured accuracy or performance.

| Asset | Type and provenance | Interpretation |
|---|---|---|
| agent-timeline | Illustrative intervals in build_visuals.py | Different sessions return during a common forecast horizon. |
| prior-continuum | Continuum Fig. 1, arXiv 2511.02230v7 | Original cache-resumption motivation, not ATFM architecture. |
| prior-thunderagent | ThunderAgent Fig. 3, arXiv 2602.13692v3 | Original program-aware architecture, not a reproduced baseline. |
| forecast-flow | fig/forecast-flow.reladraw | Implemented prediction and aggregation stages. |
| survival-example | Explicit four-duration toy distribution | Conditioning changes support and probability masses. |
| progress-concept | Linear and square-root toy curves | Reported work fraction need not equal elapsed-time fraction. |
| system | Existing fig/fig-system.reladraw | ATFM request and control paths. |
| h1-tracelab, h1-agentx | sweep_*.csv, build_paper.py | Ratios of seed-mean losses; not per-seed ratio intervals. |
| calibration | Original and calibrated h1_tracelab_r200 metrics | Coverage and loss are distinct; no across-seed CI. |
| h1b-progress | h1b_summary.csv, build_paper.py | Descriptive family scores; repeated observations are dependent. |
| regime-effects | Five base H2 paired.csv files | Paired session-bootstrap intervals versus native; not seed-level CIs. |
| h2-tradeoff | Loaded paired.csv, build_paper.py | SLO benefit and JCT cost, distinct units. |
| cap-ablation | Loaded 60 s and 600 s paired.csv means | Fixed-workload cap comparison, no inferred between-cap interval. |
| resource-costs | Loaded metrics.csv | Individual seed values and arithmetic mean, no fitted distribution. |
| research-roadmap | fig/research-roadmap.reladraw | Proposed technical dependencies, not completed results. |

`build_latex.py --figures --bundle` rebuilds figures and the portable bundle. Direct latexmk compilation uses committed PDF/PNG assets and does not need Python, Node, network access, or the raw experiment data. Vector charts use embedded fonts; the unchanged ThunderAgent raster is retained at original resolution.

## Prior-work attribution

Full image URLs, paper versions, authors, figure numbers, hashes and CC BY 4.0 links are in `sources/prior-work/manifest.json`. Captions identify each original diagram and its license. Sources are reproduced without content edits, with scaling and SVG-to-PDF conversion where applicable. Source colors and typography are preserved rather than presented as ATFM's notation.

## Quality checks

Compile with no missing citations, unresolved references, missing glyphs or overfull boxes. Render every page with Poppler and visually inspect plot labels, captions, mathematical definitions and float placement. Scientific chart axes carry units, uncertainty meaning is in captions, and all hypothetical values are explicitly identified.

## Probability expansion

`probability_visuals.py` adds six figures: hazard-residual (analytic survival conditioning), rate-posterior (fixed-prior Gamma update and inverse-rate distribution), dependence-risk (exact two-session joint-distribution comparison), demand-fan (20,000 deterministic seeded hypothetical fleet paths), sampling-loss (analytic Monte Carlo error and pinball loss), and loss-matrix (frozen original TraceLab empirical loss ratios). All except loss-matrix are explicitly illustrative. They explain implemented operations or identified assumptions, not newly measured models. Appendix C maps each addition to implementation and plan.
