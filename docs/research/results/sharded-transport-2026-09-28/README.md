# Bounded pool sharding evidence

Read the [study](../../2026-09-28-sharded-transport.md) for interpretation and limitations.

- `concurrency-{8,64,256}/`: closed-loop real-HTTP component comparisons, including
  every request and TCP trace event. All 82,944 formal requests succeeded.
- `paired/`: six baseline/candidate full-stack pairs, `bb2a561` vs `0425a58`.
  All 23,040 calls succeeded; prediction use decreased despite lower CPU/latency.
- `profile/`: separate instrumented candidate run; all 3,072 calls succeeded.
  Profile latency is not a benchmark result.
- `exploratory/`: fixed-lane initial probe; hypothesis generation without formal
  source-hash provenance, excluded from formal request totals.
- `comparison.csv`: individual paired and instrumented outcomes.
- `executed-*-driver.txt`, `profile-config.json`: exact executed scripts/config;
  adapt absolute paths and use fresh output directories.
- `pytest.txt`: final implementation, 528 passed and three skipped.
- `manifest.json`: hashes and byte lengths of 363 artifacts, plus source checks.
  Hashes describe decompressed content for gzip files; derived index/comparison
  files are not included in that artifact count.

The 64-concurrency component trial records `b7e7250` with new experiment code;
its source hashes match the subsequent `bb2a561` commit. Other source-revision
checks are explicit in the manifest. Final commit `321bf59` adds automatic stock
selection for small windows and socket tests; measured 64-slot pairs retain the
same 16-shard path. Their results remain attributed to `0425a58`.

Child exit code -15 is the harness's normal SIGTERM. Raw logs, observations,
source hashes and shutdown records are retained. These are local fake-worker
HTTP trials, not GPU throughput, TLS handshake or streaming performance evidence.
