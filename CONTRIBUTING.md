# Contributing to ATFM

Start with the [README](README.md), then read
[Working on the ATFM core](docs/development/core.md) before changing predictors,
controllers, or simulator policies.

## Set up a working copy

Use Python 3.12 and run from the repository root:

```bash
uv sync --extra dev
uv run pytest -q
```

For HTTP process development, include `--extra serve` in the sync command.
Dynamo integration and Docker collection are separate workflows documented in
[operations](docs/operations.md); they are not needed for the ordinary suite.

Create a focused branch and inspect the working tree before editing or staging:

```bash
git switch -c your-change
git status --short
```

## Make a focused change

- Keep research orchestration in `experiments/src/atfm_experiments/`. The core
  must not import the experiment package; its wheel must remain independent.
- Keep shared contracts in `schema`, model logic in `board`, decisions in
  `control`, and simulation mechanics in `sim`.
- Reuse the live board and placement-summary helpers when adding policy
  variants. Separate the source of a prediction from the action using it.
- Preserve units, horizon boundaries, random-draw order, and model fitting
  splits unless changing them is the explicit purpose of the patch.
- Keep optional service and infrastructure dependencies out of the core path.
- Document public API shapes, units, ownership, side effects, and error behavior.
  Inline comments should explain a non-obvious reason or invariant.

## Verify behavior

Run tests for affected packages during development, then run the complete suite
before submitting a behavioral change:

```bash
uv run pytest -q tests/board tests/control tests/sim
uv run pytest -q
git diff --check
```

Add regression coverage for the observable behavior being changed. Consider
empty fleets, missing telemetry, non-finite draws, exact time boundaries,
expired directives, and predictor failures where they apply. Do not replace
behavioral assertions with tests that only confirm a helper was called.

`tests/sim/golden_h2sim.json` records fixed-seed outcomes. A refactor should not
move them. If an intentional model or policy change does, explain why and review
the numerical differences before updating the fixture.

For experiment claims, provide the YAML, seed set, input provenance, metric and
comparison arm. H2 manifests record Git and input provenance; retain the exact
source/configuration used for a result. A passing Mocker check establishes
integration wiring, not real-worker performance.

## Prepare a commit or pull request

Review `git diff` and stage explicit paths. Do not include unrelated edits,
ignored trace corpora, local run directories, or generated artifacts unless
those artifacts are the purpose of the change. Inspect `git diff --cached`
before committing.

Describe the concrete problem and resulting behavior, the relevant validation,
and any remaining limitations. Update README commands and the appropriate guide
when changing setup, interfaces, or operation. Avoid adding performance claims
without a reproducible experiment and a clear comparison.
