import sys

import yaml

from atfm.experiments.h1 import H1Config, run_h1


def main(path: str) -> None:
    cfg = H1Config(**yaml.safe_load(open(path)))
    df = run_h1(cfg)
    kv = df[df["target"] == "kv_blocks"]
    print(kv.pivot_table(index=["class", "model"], columns="h", values="pinball90").round(2).to_string())


if __name__ == "__main__":
    main(sys.argv[1])
