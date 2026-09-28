# Proxy profiling evidence

See the [study](../../2026-09-27-proxy-profiling.md) for methods and conclusions.

- `exploratory-cprofile/`: retained initial profiles; not used for CPU attribution.
- `cpu-before/`, `cpu-after/`: Yappi CPU profiles without/with sniffio.
- `unprofiled/`: twelve latency/CPU comparison cases, three seeds per control mode and environment.
- `reproduction/`: exact scripts, configurations, package inventories, and focused-test results.
- `comparison.csv`: derived case-level metrics, with profiling explicitly labeled.
- `manifest.json`: uncompressed length and SHA-256 for 384 retained source files.

Function CSVs retain thread identity. Binary pstats merge threads; load only
trusted profile files. Gzip files decompress to their original bytes. Times from
instrumented and uninstrumented runs must not be combined. These are local fake
worker tests, not real model serving capacity or scheduling quality results.
