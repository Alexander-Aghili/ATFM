# Offline tuning workflow evidence

Source revision: `d283acd`. See the [study](../../2026-09-28-policy-tuning.md)
for interpretation and limitations. This is functional smoke evidence, not a
production performance result. Documentation builds overlapped part of the run.

`smoke/` contains the predeclared plan, environment record, randomized schedules,
frozen search selection, held-out decision and 24 case directories. All 384
requests succeeded. `pytest.log.gz` records 555 passing tests and three skips;
`executed-smoke.log.gz` records execution order and feasibility outcomes.

`manifest.json` records SHA-256 and byte counts for 489 evidence files. For `.gz`
files, decompress before hashing. The manifest and this explanatory README are
not self-hashed. Original paths inside trial records identify ignored working
run directories; the same directory structure is preserved below `smoke/`.
The result never deploys or changes runtime settings.
