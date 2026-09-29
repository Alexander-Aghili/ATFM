# Proxy ranking and transport evidence

Read the [study](../../2026-09-28-proxy-ranking.md) for methods, interpretation and limits.

- `ranks/`: six unprofiled baseline/candidate pairs, `f180401` vs `3084f34`.
- `pool/`: six separate pairs, `f180401` vs rejected `14cc14a`.
- `profiles/`: discovery (`36f0860`), rejected pool, and retained rank profiles.
  The pool profile is incomplete: 3,069 successes, one failure, two unissued turns.
- `comparison.csv`: per-case measurements; instrumented and uninstrumented groups
  must not be combined for latency analysis.
- `microbenchmark.json`, `executed-microbenchmark.txt`: exact query comparison;
  construction and mutation excluded from timing.
- `executed-*-driver.txt`, `profile-config.json`: executed commands/configuration;
  adapt absolute paths and use fresh output directories to reproduce.
- `pytest.txt`: final retained implementation test run.
- `manifest.json`: revisions and hashes of 559 archived artifacts. Hashes and byte
  counts refer to decompressed contents for `.gz` files. Derived comparison and
  index files are outside that artifact list.

Each trial includes source hashes, configuration, environment, client requests,
proxy traces, control events, observations, shutdown status and logs. Large
records and pstats/profile text use gzip without changing their contents. Child
exit code -15 records harness SIGTERM. All 46,080 unprofiled calls succeeded;
profiling outcomes are reported separately. These are local fake-worker trials,
not real-model throughput or GPU/cache validation.
