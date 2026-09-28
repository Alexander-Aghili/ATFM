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
    a = _arguments()
    train = TraceTable.from_parquet(a.train)
    fc = SessionForecaster(ProgressPredictor().fit(train), ExogenousModel().fit(train),
                           horizons=[10.0, 30.0, 120.0, 300.0, 900.0], n=256)
    board = LiveBoard(SessionRegistry(), fc, tick_s=a.tick)
    rng = np.random.default_rng(0)
    if a.serve is not None:
        _serve(a, board, rng)
        return
    _file_loop(a, board, rng)


def _file_loop(a, board, rng):
    seen = 0
    with open(a.snapshots, "a") as out:
        while True:
            ev = read_events(a.events)
            for e in ev[seen:]:
                board.registry.apply(e)
            seen = len(ev)
            now = time.time()
            snap = board.step(now, rng)
            _write_snapshot(out, now, snap)
            if a.once:
                break
            time.sleep(a.tick)


def _write_snapshot(out, now, snap):
    rec = {"t": now, "model_id": snap.model_id, "horizons": snap.horizons,
           "q50": {t: {c: snap.quantiles(t, c, 0.5).tolist() for c in CLASSES} for t in TARGETS},
           "q90": {t: {c: snap.quantiles(t, c, 0.9).tolist() for c in CLASSES} for t in TARGETS},
           "endogenous_fraction": {c: snap.endogenous_fraction[c].tolist() for c in CLASSES}}
    out.write(json.dumps(rec) + "\n")
    out.flush()


def _serve(a, board, rng):
    import uvicorn
    from atfm.board.service import create_board_app
    from atfm.bus import JsonlBus
    app = create_board_app(board, bus=JsonlBus(a.events), rng=rng)
    if a.control:
        _attach_scraper(a, app)
    uvicorn.run(app, host="127.0.0.1", port=a.serve, log_level="warning")
    return


def _attach_scraper(a, app):
    import threading
    import yaml
    from atfm.board.service import configure_controllers
    ccfg = yaml.safe_load(open(a.control)) or {}
    scrape = configure_controllers(app, ccfg)
    interval = float(ccfg.get("metrics", {}).get("interval_s", a.tick))

    def loop():
        while True:
            scrape()
            time.sleep(interval)
    threading.Thread(target=loop, daemon=True).start()


def _arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--snapshots", required=True)
    ap.add_argument("--train", required=True, help="parquet trace table to fit the predictor on")
    ap.add_argument("--tick", type=float, default=5.0)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--serve", type=int, default=None, help="serve the board HTTP API on this port instead of the file loop")
    ap.add_argument("--control", default=None, help="YAML with gdp/touch/tier/replica/metrics sections to attach controllers")
    a = ap.parse_args()
    return a


if __name__ == "__main__":
    main()
