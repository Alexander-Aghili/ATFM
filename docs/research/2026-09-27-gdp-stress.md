# GDP stress tests: 100,000 and one million synthetic sessions

This extends the [large-control-path study](2026-09-27-large-control-paths.md)
to substantially larger inputs. These measurements exercise the actual optimized
`GdpPlanner` on generated forecast samples and session demands. They do not run
a million agents, a forecast model, HTTP requests, or GPU inference.

## Results

| Sessions | Slots | Samples | Regime | Wall time | CPU time | Peak RSS |
| ---: | ---: | ---: | --- | ---: | ---: | ---: |
| 100,000 | 10,000 | 1,024 | saturated | 7.32 s | 7.32 s | 645 MiB |
| 100,000 | 10,000 | 1,024 | mixed | 4.71 s | 4.71 s | 764 MiB |
| 1,000,000 | 300 | 128 | saturated | 15.91 s | 15.91 s | 2,552 MiB |

Every row is a separate process with one untimed warmup and three timed runs.
The table reports median planning wall time and process CPU time. Peak RSS is
the process high-water mark, including imports, fixture construction, planning,
result objects, and result hashing. It is not incremental planner allocation or
an estimate of full-service memory. The benchmark may retain the prior result
while allocating the next one. MiB is 2^20 bytes.

The 100,000-session / 10,000-slot case has about 41 times the session-slot
product of the previous 8,192-session / 3,000-slot case. The one-million-session
case increases session count another tenfold while keeping fewer slots, to
separate session/output overhead from very wide slot search.

The saturated 100,000-session case takes 7.32 seconds per plan and therefore
cannot sustain a five-second planning interval on one serial planner. The mixed
case takes 4.71 seconds, leaving little margin for forecasting, ingestion,
directive delivery, or scheduling jitter. This is concrete evidence that CPU
work is consequential at this scale. Similar CPU and wall times indicate
CPU-dominated planning in these isolated component runs; they say nothing about
CPU versus network or KV movement in a complete deployment.

## Synthetic workload

- Each slot is one second. The horizon has one forecast point per slot and the
  maximum hold covers the whole horizon, deliberately exposing long searches.
- Cumulative interactive samples are a cumulative sum of independent uniform
  per-slot values between 800 and 1,000, using seed 7. Both modeled resources
  share the input sample array; derived slot data is prepared separately. This
  is a computational fixture with perfectly correlated resource demands, not
  a calibrated serving trace.
- In the saturated regime, every session starts at ETA zero and requests ten
  units of each resource; capacity is 500. All candidate slots are infeasible
  and each session receives a capped directive. This stresses failed searches.
- In the mixed regime, ETA varies across the horizon, demands independently vary
  from 1 through 99 for each resource, and sessions span 16 tenants. Capacity
  is 1,000. Sequential commitments influence later feasibility; no tenant
  overrides are supplied. Heterogeneity does not make this a production trace.
- Sorting, slot interpolation, threshold preparation, greedy assignment, and
  directive construction are timed. Setup and result hashing are excluded from
  timing. Every measured result must match the warmup's SHA-256 fingerprint.

The driver streams canonical JSON into the fingerprint hash rather than
allocating an additional full result tree. Regression tests verify equivalence
with the former whole-list JSON hash and repeatability of all three regimes.
No full S-by-F or S-by-F-by-D tensor is allocated by the planner; search still
performs O(S F) work across the sessions.

## Consequences for the design

The exact threshold optimization removed the repeated sample-count multiplier;
it did not remove the session-slot search product or the O(S) directive output.
The first priorities suggested by this evidence are to bound the planning
horizon/resolution to the control need, investigate indexed feasible-slot search
and incremental planning, and move long planning work out of the request event
loop. Coarser slots change decision resolution and must be evaluated as such.
Partitioning by independent worker/capacity pools may help, but arbitrary
session sharding can violate shared-capacity constraints.

Rust could reduce constants in a measured search/output kernel. A wholesale
rewrite does not eliminate S times F work or the cost of creating and delivering
one directive per session. The next experiment should measure the full control
cycle and outcome quality under representative workloads before choosing a
language boundary or claiming production capacity.

## Reproduce

Run sequentially, each command as its own process:

```bash
uv run python -m atfm_experiments.benchmark_gdp \
  --sessions 100000 --slots 10000 --draws 1024 --regimes saturated \
  --out runs/gdp-stress-100k
uv run python -m atfm_experiments.benchmark_gdp \
  --sessions 100000 --slots 10000 --draws 1024 --regimes mixed \
  --out runs/gdp-stress-mixed
uv run python -m atfm_experiments.benchmark_gdp \
  --sessions 1000000 --slots 300 --draws 128 --regimes saturated \
  --out runs/gdp-stress-million
```

The benchmark accepts independent session, slot, and draw lists and
`--regimes open saturated mixed`. RSS is cumulative when several cases share a
process, so use one case per invocation for memory comparisons. Three
repetitions are local measurements, not tail-latency estimates or confidence
intervals. Hardware was the same Intel Core Ultra 9 185H workstation as the
preceding study; Python 3.12.13 and NumPy 2.5.3. CPU affinity and numerical-library
thread counts were not pinned and background workstation activity was not
controlled. The recorded source and driver hashes identify the exact code.

Artifacts:

- [All results](results/gdp-stress-2026-09-27/comparison.csv).
- 100,000 saturated: [timings](results/gdp-stress-2026-09-27/100k.csv),
  [environment](results/gdp-stress-2026-09-27/100k-environment.json).
- 100,000 mixed: [timings](results/gdp-stress-2026-09-27/mixed.csv),
  [environment](results/gdp-stress-2026-09-27/mixed-environment.json).
- One million saturated: [timings](results/gdp-stress-2026-09-27/million.csv),
  [environment](results/gdp-stress-2026-09-27/million-environment.json).
