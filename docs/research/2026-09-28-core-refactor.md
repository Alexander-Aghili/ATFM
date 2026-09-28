# Core refactor validation, 28 September 2026

This maintenance pass consolidates sidecar configuration, bounded gate waiting,
command execution, simulator callbacks, admission-index methods, trace scalar
sentinels, and forecast axis names. Existing configuration imports remain valid.
The net change in `src/atfm` is 54 fewer lines against `c866cc1`; tests and
explanatory documentation are additional. No scheduling algorithm is changed.

## Design choices

The generic executor owns launching and output conversion; the mini-SWE-agent
mixin adds its completion hook. Local and Docker environments share constructor
setup while retaining their distinct command/environment construction. Generic
adapters can now import configuration without importing an optional harness.

Simulator policies inherit identical callbacks and index calculations. Placement
variants retain their overridden expected-duration term. Missing scalar detection
keeps its narrow None/float-NaN semantics. Forecast axis constants have one schema
owner and remain available through their former importing modules.

We rejected a shared per-line parser helper: its interleaved microbenchmark
increased wrapped parsing median from 29.38 to 30.82 ms (4.9%). Both parser loops
remain inline. A progress match stops parser traversal even when duplicated;
a data match does not. Bus failures must not create duplicate progress attempts.
These contracts are covered by regression tests, along with optional-import
independence and explicitly supplied falsey command builders.

## Validation and timing

The final full suite in the existing development/harness environment passed:
**420 passed, 5 skipped, 6 warnings**. Warnings concern an upstream AnyIO alias
and existing empty/NaN statistical fixtures. Skips remain skips, not evidence
that optional GPU integrations were exercised.

| Workload | Baseline median | Candidate median | Parity |
| --- | ---: | ---: | --- |
| Wrapped output, 10,000 progress lines | 28.16 ms | 28.67 ms | Identical output/event hash |
| Subprocess, 10,000 progress lines | 46.46 ms | 46.41 ms | Identical output/event hash |
| Forecast, 8,192 sessions | 258.11 ms | 251.75 ms | Identical result hash |
| Ten simulation policy variants, 120 sessions each | 9.19–66.81 ms | 8.96–66.66 ms | All ten hashes identical |

Sidecar measurements use three sequential baseline/candidate blocks, seven
measured repetitions after one warmup per mode/block (21 samples per side).
Each produces 10,002 events. Hashes exclude random call IDs and include all other
event fields and output bytes. Policy measurements use one warmup and three
repetitions; hashes include request rows, final RNG state, and session logs.
Forecast measurements use three repetitions after warmup. Setup and hashing
are outside the timed regions except for work explicitly inside the scripts.

These short CPU trials establish output parity for the fixtures and show no
large retained regression; they cannot establish identical latency on arbitrary
hardware or workloads. The wrapped difference is +1.8%, subprocess -0.1%, and
forecast -2.5%. These are observations, not claimed speedups. Complexity and
allocation strategy in the forecast/scheduling hot paths remain unchanged.

## Reproduction and provenance

[Raw evidence and scripts](results/core-refactor-2026-09-28/) retain all samples.
The common interpreter was `/tmp/atfm-proxy-sniffio/bin/python` (Python 3.12.13),
with NumPy 2.5.3 and pandas 3.0.6 on Linux x86-64. Baseline source was extracted
with `git archive c866cc1 src/atfm` and selected using `PYTHONPATH`; both sides
used the same interpreter and machine. Benchmark processes ran sequentially.

Run `sidecar_benchmark.py OUTPUT.json` and `policy_benchmark.py OUTPUT.json`
with `PYTHONPATH` pointing to the source under test. For forecast, use
`python -m atfm_experiments.benchmark_cpu --cases forecast --sizes 8192 --repeats 3 --out DIR`.
The retained sidecar files are explicitly named `sidecar-retained-*`;
`rejected-parser-*` records the abandoned function extraction. Policy results
precede the equivalent scalar-helper relocation; forecast results follow that
relocation but precede reverting the unrelated sidecar helper. The forecast
environment manifests therefore include that intermediate sidecar source, and
their checkout SHA is not the source-selection mechanism for baseline runs.
The final suite covers all retained changes together.
