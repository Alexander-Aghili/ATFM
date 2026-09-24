import argparse
import random
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=600)
    ap.add_argument("--rate", type=float, default=10.0)
    ap.add_argument("--signal", choices=["strong", "weak", "none"], default="strong")
    a = ap.parse_args()
    done = 0
    stage = 0
    while done < a.rows:
        time.sleep(1.0)
        done = min(a.rows, done + int(a.rate * random.uniform(0.8, 1.2)))
        if a.signal == "strong":
            print(f"processed {done}/{a.rows} rows", flush=True)
        elif a.signal == "weak":
            q = int(4 * done / a.rows)
            if q > stage:
                stage = q
                print(f"stage {stage}/4", flush=True)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
