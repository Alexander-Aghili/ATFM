"""Run the control loop against a board service and a proxy.

    uv run python scripts/run_control.py --board http://127.0.0.1:8081 --proxy http://127.0.0.1:8799 --interval 5
"""
import argparse

from atfm.control.loop import ControlLoop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="http://127.0.0.1:8081")
    ap.add_argument("--proxy", default="http://127.0.0.1:8799")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--log", default="runs/control/control.jsonl")
    ap.add_argument("--steps", type=int, default=None)
    a = ap.parse_args()
    ControlLoop(a.board, a.proxy, interval_s=a.interval, log_path=a.log).run(steps=a.steps)


if __name__ == "__main__":
    main()
