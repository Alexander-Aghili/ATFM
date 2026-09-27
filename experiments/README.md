# Experiment workspace

This directory contains research configurations and the separately packaged
`atfm_experiments` runners. The reusable library remains in `../src/atfm/`.
Experiment orchestration depends on the core; the core must not depend on these
runners.

| Path | Purpose |
| --- | --- |
| `src/atfm_experiments/h1.py` | H1 configuration, fitting, calibration, scoring, and result output. |
| `src/atfm_experiments/h2sim.py` | H2 workloads, policy selection, simulation runs, comparisons, and provenance. |
| `*.yaml` | Experiment and local integration configurations. |
| `pyproject.toml` | Independent `atfm-experiments` distribution, part of the root uv workspace. |

From the repository root:

```bash
uv sync --extra dev
uv run python scripts/run_h2sim.py experiments/quickstart.yaml
uv run python scripts/run_h1.py experiments/h1_synthetic.yaml
```

The root's default `dev` dependency group installs the experiment package in
editable mode. The existing `dev` extra supplies test tools. Commands and YAML
paths are unchanged by the package split; Python imports now use
`atfm_experiments.h1` and `atfm_experiments.h2sim` instead of `atfm.experiments`.
There is intentionally no compatibility module inside the core package.

For a core-only service environment, exclude development groups:

```bash
uv sync --no-dev --extra serve
```

This omits the experiment package and is not an environment for running the full
repository test suite. Restore `uv sync --extra dev` for development. Building
the core wheel includes only `atfm`; the experiment package has its own wheel:

```bash
uv build --package atfm
uv build --package atfm-experiments
```

See the [experiment overview](../README.md#experiments),
[core development guide](../docs/development/core.md), and
[research index](../docs/research/README.md) for model contracts and evidence.
