from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from atfm.board.calibrate import CalibratedForecaster, fit_inflation
from atfm.board.forecaster import CLASSES, TARGETS, ExogenousModel, SeriesForecaster, SessionForecaster
from atfm.board.predictors import (BackendPredictor, ConstantSeries, HistoryPredictor, KalmanSeries,
                                   ProgressPredictor, SurvivalPredictor)
from atfm.board.replay import FleetReplayer
from atfm.eval.forecast import aggregate_scores, score_tick, surge_events
from atfm.schema.trace import TraceTable
from atfm.traces.synthetic import WorkloadSpec, generate
from atfm.traces.transform import overlay_sessions, split_by_session_families, split_by_time_blocks


class H1Config(BaseModel):
    name: str
    source: Literal["tracelab", "synthetic", "sidecar"]
    tracelab_parquet: str | None = None
    sidecar_events: str | None = None
    synthetic: WorkloadSpec | None = None
    overlay_rate_per_hour: float | None = None
    overlay_duration_s: float = 4 * 3600.0
    tick_s: float = 30.0
    horizons: list[float] = Field(default_factory=lambda: [10.0, 30.0, 120.0, 300.0, 900.0])
    n_samples: int = 512
    models: list[str] = Field(default_factory=lambda: ["B0", "B1", "B2", "M1", "M2", "M3"])
    block_seconds: float = 7 * 86400.0
    split: Literal["time", "session"] = "time"   # "session" holds out whole families (for per-trace-relative timestamps)
    test_fraction: float = 0.3
    seed: int = 0
    capacity_quantile: float = 0.95
    calibrate: bool = False          # fit per-horizon dispersion inflation out-of-sample within train (session models only)
    calibration_ticks: int = 200
    calibration_holdout: float = 0.5  # share of train (by time block or family) held out to fit the inflation on
    out_dir: str = "runs"


def _tables(cfg: H1Config) -> tuple[TraceTable, TraceTable]:
    if cfg.source == "synthetic":
        assert cfg.synthetic is not None
        train = generate(cfg.synthetic.model_copy(update={"seed": cfg.synthetic.seed + 1000}))
        test = generate(cfg.synthetic)
        return train, test
    if cfg.source == "sidecar":
        from atfm.bus import read_events
        from atfm.traces.sidecar import events_to_trace_table
        assert cfg.sidecar_events is not None
        table = events_to_trace_table(read_events(cfg.sidecar_events))
        return split_by_time_blocks(table, cfg.block_seconds, cfg.test_fraction, cfg.seed)
    assert cfg.tracelab_parquet is not None
    table = TraceTable.from_parquet(cfg.tracelab_parquet)
    if cfg.split == "session":
        train, test = split_by_session_families(table, cfg.test_fraction, cfg.seed)
    else:
        train, test = split_by_time_blocks(table, cfg.block_seconds, cfg.test_fraction, cfg.seed)
    if cfg.overlay_rate_per_hour:
        # Predictors fit on the real train sessions; only the test fleet is replayed as a Poisson overlay.
        test = overlay_sessions(test, cfg.overlay_rate_per_hour, cfg.overlay_duration_s, cfg.seed + 2)
    return train, test


def _session_starts(df: pd.DataFrame) -> pd.DataFrame:
    """Exogenous arrivals: first calls of root sessions. Children are endogenous (forecast via spawn)."""
    root = df["parent_session_id"].isna()
    return df[(df["turn_index"] == 0) & root][["t_request", "class"]].sort_values("t_request")


def _build(cfg: H1Config, train: TraceTable) -> dict:
    # Each session model owns its ExogenousModel: they are updated once per model per tick.
    def exo():
        return ExogenousModel().fit(train)
    reg = {
        "B0": lambda: SeriesForecaster(ConstantSeries(), cfg.horizons, cfg.n_samples),
        "B1": lambda: SeriesForecaster(KalmanSeries(), cfg.horizons, cfg.n_samples),
        "B2": lambda: SessionForecaster(HistoryPredictor().fit(train), exo(), cfg.horizons, cfg.n_samples),
        "M1": lambda: SessionForecaster(SurvivalPredictor().fit(train), exo(), cfg.horizons, cfg.n_samples),
        "M2": lambda: SessionForecaster(ProgressPredictor().fit(train), exo(), cfg.horizons, cfg.n_samples),
        "M3": lambda: SessionForecaster(BackendPredictor().fit(train), exo(), cfg.horizons, cfg.n_samples),
    }
    unknown = [m for m in cfg.models if m not in reg]
    if unknown:
        raise ValueError(f"unknown models {unknown}; choose from {sorted(reg)}")
    return {m: reg[m]() for m in cfg.models}


def _ticks(t_min: float, t_max: float, window_s: float, horizons: list[float], tick_s: float) -> np.ndarray:
    """Ticks cover the fleet window only; sessions may run past it but are not scored there."""
    end = min(t_max, t_min + window_s) - max(horizons)
    return np.arange(t_min, end + 1e-9, tick_s)


def _calibrate(cfg: H1Config, models: dict, train: TraceTable) -> dict:
    """Wrap session forecasters with a dispersion factor fitted on ticks of the train table.

    Fitting is out-of-sample within train: the predictor is fitted on one part of train and the inflation
    on ticks of the other part, so the factor reflects the generalization gap the test run will meet.
    Fresh model instances are used so no train-fleet state leaks into the forecasters that score the test."""
    if cfg.calibration_holdout <= 0.0:  # in-sample (diagnostic only)
        fit_part, cal_part = train, train
    elif cfg.source == "synthetic" or cfg.split == "time":
        fit_part, cal_part = split_by_time_blocks(train, cfg.block_seconds if cfg.source != "synthetic" else
                                                  max(60.0, (train.time_range()[1] - train.time_range()[0]) / 8),
                                                  cfg.calibration_holdout, cfg.seed + 11)
    else:
        fit_part, cal_part = split_by_session_families(train, cfg.calibration_holdout, cfg.seed + 11)
    if len(fit_part) == 0 or len(cal_part) == 0:
        return {}
    fitting = _build(cfg, fit_part)
    if cfg.source != "synthetic" and cfg.overlay_rate_per_hour:
        # The test is a Poisson fleet at this rate; calibrate on the same kind of fleet, not on sparse calendar time.
        cal_part = overlay_sessions(cal_part, cfg.overlay_rate_per_hour, cfg.overlay_duration_s, cfg.seed + 12)
    rep = FleetReplayer(cal_part)
    t_min, t_max = cal_part.time_range()
    if cfg.source == "synthetic" and cfg.synthetic:
        t_max = min(t_max, t_min + cfg.synthetic.duration_s)
    elif cfg.overlay_rate_per_hour:
        t_max = min(t_max, t_min + cfg.overlay_duration_s)   # score the fleet window, not the longest session's tail
    end = t_max - max(cfg.horizons)
    if end <= t_min:
        return {}
    ticks = list(np.linspace(t_min, end, num=min(cfg.calibration_ticks, max(2, int((end - t_min) // cfg.tick_s) + 1))))
    starts = _session_starts(cal_part.df)
    starts_t = starts["t_request"].to_numpy(float)
    starts_c = starts["class"].tolist()
    factors = {}
    for name, fc in list(models.items()):
        if not isinstance(fc, SessionForecaster):
            continue
        fit_fc = fitting[name]
        ptr = [0]

        def feed(t, fit_fc=fit_fc, ptr=ptr):  # same incremental arrival feed as the scored run
            new_ptr = int(np.searchsorted(starts_t, t, side="right"))
            fit_fc.exo.update(t, [(float(starts_t[i]), starts_c[i]) for i in range(ptr[0], new_ptr)])
            ptr[0] = new_ptr

        k = fit_inflation(fit_fc, ticks, truth_fn=lambda t: rep.demand_truth(t, cfg.horizons), target=0.9,
                          rng=np.random.default_rng(cfg.seed + 7), states_fn=rep.states_at, on_tick=feed)
        models[name] = CalibratedForecaster(fc, k)
        factors[name] = {c: v.tolist() for c, v in k.items()}
    return factors


def _perturbed_at(cfg: H1Config, t: float) -> bool:
    if cfg.source != "synthetic" or cfg.synthetic is None:
        return False
    return any(p.t_start <= t < p.t_end for p in cfg.synthetic.perturbations)


def run_h1(cfg: H1Config) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    train, test = _tables(cfg)
    if len(train) == 0:
        raise ValueError("train split is empty: use smaller block_seconds or a lower test_fraction for this table")
    if len(test) == 0:
        raise ValueError("test split is empty: use a higher test_fraction or more data")
    models = _build(cfg, train)
    factors = _calibrate(cfg, models, train) if cfg.calibrate else {}
    rep = FleetReplayer(test)
    t_min, t_max = test.time_range()
    if cfg.source == "synthetic" and cfg.synthetic:
        window = cfg.synthetic.duration_s
    elif cfg.source == "sidecar":
        window = t_max - t_min  # the collection is already a fleet; score its whole span
    else:
        window = cfg.overlay_duration_s
    ticks = _ticks(t_min, t_max, window, cfg.horizons, cfg.tick_s)
    if len(ticks) == 0:
        raise ValueError(f"test span ({min(t_max - t_min, window):.0f} s) is shorter than the longest horizon "
                         f"({max(cfg.horizons):.0f} s): no tick can be scored; shorten the horizons or collect more")
    df = test.df
    ended = df[df["t_tool_end"].notna()][["t_tool_end", "backend_id", "tool_name", "t_tool_start"]].sort_values("t_tool_end")
    ended_t = ended["t_tool_end"].to_numpy(float)
    starts = _session_starts(df)
    starts_t = starts["t_request"].to_numpy(float)
    records = []
    series = {m: {"truth": [], "q90": []} for m in models}
    k300 = cfg.horizons.index(300.0) if 300.0 in cfg.horizons else len(cfg.horizons) - 1
    exo_ptr = end_ptr = 0
    for t in ticks:
        t = float(t)
        truth = rep.demand_truth(t, cfg.horizons)
        states = rep.states_at(t)
        new_ptr = int(np.searchsorted(starts_t, t, side="right"))
        new_starts = [(float(starts_t[i]), starts["class"].iloc[i]) for i in range(exo_ptr, new_ptr)]
        exo_ptr = new_ptr
        e_ptr = int(np.searchsorted(ended_t, t, side="right"))
        completions = ended.iloc[end_ptr:e_ptr]
        end_ptr = e_ptr
        pert = _perturbed_at(cfg, t)
        for name, fc in models.items():
            inner = getattr(fc, "inner", fc)  # a CalibratedForecaster wraps the real one
            if isinstance(inner, SessionForecaster):
                inner.exo.update(t, new_starts)
                if isinstance(inner.predictor, BackendPredictor):
                    for r in completions.itertuples():
                        inner.predictor.observe_completion(r.backend_id, r.tool_name, float(r.t_tool_end - r.t_tool_start))
            snap = fc.forecast(t, states, rng)
            for tgt in TARGETS:
                for c in CLASSES:
                    records.append(score_tick(t, snap.model_id, tgt, c, cfg.horizons, snap.samples[tgt][c],
                                              truth[tgt][c], pert, snap.endogenous_fraction[c]))
            series[name]["truth"].append(float(sum(truth["kv_blocks"][c][k300] for c in CLASSES)))
            series[name]["q90"].append(float(np.quantile(snap.total("kv_blocks")[k300], 0.9)))
            if isinstance(inner, SeriesForecaster):
                inner.observe(t, truth)
    metrics = aggregate_scores(records)
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "metrics.csv", index=False)
    surge = {}
    for name in models:
        truth_s = np.asarray(series[name]["truth"])
        q90 = np.asarray(series[name]["q90"])
        cap = float(np.quantile(truth_s, cfg.capacity_quantile)) if len(truth_s) else 0.0
        ev = surge_events(q90, truth_s, cap, cfg.tick_s, horizon_s=300.0)
        surge[name] = {"capacity": cap, **ev}
    (out / "surge.json").write_text(json.dumps(surge, indent=2))
    (out / "config.json").write_text(cfg.model_dump_json(indent=2))
    if cfg.calibrate:
        (out / "calibration.json").write_text(json.dumps({"horizons": cfg.horizons, "inflation": factors}, indent=2))
    return metrics
