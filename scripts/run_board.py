import argparse
import json
import time

import numpy as np

from atfm.board.forecaster import CLASSES, TARGETS, ExogenousModel, SessionForecaster
from atfm.board.live import LiveBoard, SessionRegistry
from atfm.board.predictors import ProgressPredictor
from atfm.bus import read_events
from atfm.schema.trace import TraceTable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--train", required=True, help="parquet trace table to fit the predictor on")
    ap.add_argument("--tick", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    train = TraceTable.from_parquet(a.train)
    fc = SessionForecaster(ProgressPredictor().fit(train), ExogenousModel().fit(train),
                           horizons=[10.0, 30.0, 120.0, 300.0, 900.0], n=256)
    board = LiveBoard(SessionRegistry(), fc, tick_s=a.tick)
    rng = np.random.default_rng(0)
    seen = 0
    with open(a.snapshots, "a") as out:
        while True:
            ev = read_events(a.events)
            for e in ev[seen:]:
                board.registry.apply(e)
            seen = len(ev)
            now = time.time()
            snap = board.step(now, rng)
            rec = {"t": now, "model_id": snap.model_id, "horizons": snap.horizons,
                   "q50": {t: {c: snap.quantiles(t, c, 0.5).tolist() for c in CLASSES} for t in TARGETS},
                   "q90": {t: {c: snap.quantiles(t, c, 0.9).tolist() for c in CLASSES} for t in TARGETS},
                   "endogenous_fraction": {c: snap.endogenous_fraction[c].tolist() for c in CLASSES}}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            if a.once:
                break
            time.sleep(a.tick)


if __name__ == "__main__":
    main()
