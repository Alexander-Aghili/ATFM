"""Write the board's training table with the replayed evaluation roots removed (no train/test leakage)."""
import argparse
import hashlib
import json
from pathlib import Path

from atfm.schema.trace import TraceTable
from atfm.traces.agentx import exclude_traces


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="data/agentx/agentx.parquet")
    ap.add_argument("--selection", required=True, help="selection.json from gpu_cache.trace_selection")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    held_out = [case["trace_id"] for case in json.loads(Path(a.selection).read_text())["cases"].values()]
    table = TraceTable.from_parquet(a.parquet)
    kept = exclude_traces(table, held_out)
    kept.to_parquet(a.output)
    record = dict(source=a.parquet, source_sha256=hashlib.sha256(Path(a.parquet).read_bytes()).hexdigest(),
                  held_out=held_out, rows_in=len(table.df), rows_out=len(kept.df),
                  output_sha256=hashlib.sha256(Path(a.output).read_bytes()).hexdigest())
    Path(a.output).with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(record))


if __name__ == "__main__":
    main()
