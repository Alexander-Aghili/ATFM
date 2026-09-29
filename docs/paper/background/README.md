# LLM inference serving: technical background

Read [the paper](llm-serving-background.pdf) for a tutorial and research survey
covering attention/KV memory, vLLM and other engines, LMCache and cache tiers,
disaggregated serving, agent scheduling, and ATFM's specific research hypothesis.
The source cutoff is **28 September 2026**; the ATFM code baseline is `900d63b`, with the prediction-overload implementation updated through `40602f3` and board isolation through `2bc3f8b`.

This is a separate background paper, not a replacement for the main ATFM research
manuscript. It distinguishes published mechanisms, derived examples, local
prototype evidence, and unvalidated real-worker claims. [SOURCES.md](SOURCES.md)
records the primary sources and the scope of each; mutable documentation is not
a pinned release contract.

## Rebuild

From this directory, with TeX Live and latexmk installed:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error llm-serving-background.tex
```

Or from the repository root, using the checked-in vector figures:

```bash
python3 docs/paper/background/build.py
```

To regenerate all original figures and the portable source bundle:

```bash
.venv/bin/python docs/paper/background/build.py --figures --bundle
```

Figure regeneration needs NumPy, Matplotlib, Node/npx with reladraw, and Inkscape.
A normal PDF build uses only Python's standard library plus latexmk/TeX Live.
The wrapper builds in `tmp/pdfs/llm-serving-background`, rejects unresolved
references and overflowing text, and copies the PDF here and to `output/pdf/`.
It does not overwrite `docs/paper/atfm-paper.pdf` or its manuscript.

## Editing and evidence

- `sections/` contains the editable prose and native LaTeX equations/tables.
- `figures/*.reladraw` contains architecture sources; SVG/PDF are generated.
- `plots.py` produces analytical plots. All numbers are illustrative assumptions,
  recorded in `illustration-inputs.json`, not GPU or ATFM benchmark results.
- `make_sources.py` owns the bibliography metadata and regenerates
  `sources.json`, `SOURCES.md`, and `references.tex`.
- `build.py --bundle` creates an Overleaf-ready ZIP with pre-rendered figures.
  Select `llm-serving-background.tex` and pdfLaTeX.

When updating the survey, revise the cutoff and edition/scope notes, check the
source rather than copying a headline speedup, and distinguish an accepted
control request from completed movement and observed cache reuse. Review the
PDF after changes; a clean LaTeX log does not establish visual quality.

## Reused paper artwork

Five figures reproduce original artwork from vLLM (Figure 6), LMCache (Figure 5),
DistServe (Figure 6), Continuum (Figure 1), and ThunderAgent (Figure 3). The last
two reuse the repository's existing archive. Captions identify the authors,
paper version, figure number, and license. See [the artwork manifest](figures/papers/manifest.json)
for source URLs, SHA-256 hashes, modifications, and license links.

The artwork is unchanged apart from scaling and SVG-to-PDF conversion. DistServe
artwork retains CC BY-SA 4.0; the other reproduced artwork retains CC BY 4.0.
These third-party licenses apply to their respective figures. Source SVG/PNG
files and the PDFs used for typesetting are included in the standalone bundle.
To regenerate an SVG's PDF, run `inkscape SOURCE.svg --export-type=pdf`.
Four original synthesis diagrams and four analytical plots complement the
reproductions where a project-specific explanation is needed.

The ATFM evaluation section incorporates the [proxy ranking and transport study](../../research/2026-09-28-proxy-ranking.md), distinguishing component scaling from end-to-end latency. This is a local implementation update, not a new literature survey.

The ATFM section now distinguishes the rejected larger shared HTTP pool from the subsequent [bounded sharding study](../../research/2026-09-28-sharded-transport.md). The literature-review cutoff is unchanged.

The ATFM section also introduces the [offline tuning protocol](../../development/policy-tuning.md), distinguishing a context-specific best observed configuration from a universal optimum.
