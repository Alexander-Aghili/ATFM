import sys

import pandas as pd
import yaml

from atfm.experiments.h2sim import H2SimConfig, run_h2sim

if __name__ == "__main__":
    cfg = H2SimConfig(**yaml.safe_load(open(sys.argv[1])))
    df = run_h2sim(cfg)
    cols = ["slo_attainment_calls", "ttft_after_tool_p95", "bg_jct_mean", "deadline_hit_rate", "recomputed_prefill_tokens",
            "hold_kv_block_s", "mean_held_s_background", "caps", "queue_proxy_share"]
    print(df.groupby("arm")[cols].mean().round(3).to_string())
    print(pd.read_csv(f"{cfg.out_dir}/{cfg.name}/paired.csv").round(3).to_string())
