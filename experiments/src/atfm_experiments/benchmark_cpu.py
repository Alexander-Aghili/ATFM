"""Reproducible CPU scaling trials; timings exclude setup and profiling overhead."""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
import pstats
import subprocess
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

from atfm.board.predictors.duration import DurationModel
from atfm.eval.serving import paired_bootstrap
from atfm.sim.engine import EngineConfig, Request, Worker
from atfm.traces.synthetic import ClassSpec, ToolSpec, WorkloadSpec
from atfm_experiments.h1 import H1Config, run_h1
from atfm_experiments.h2sim import H2SimConfig, run_h2sim


def make_case(case: str, size: int, out: Path):
    """Return a repeatable workload; ``size`` has the case-specific units below."""
    if case == "eviction":
        def run():
            worker = Worker("w", EngineConfig(size, 1, 20000.0, 40.0))
            worker.resident = OrderedDict((f"s{i}", 1) for i in range(size))
            return worker._make_room(size // 2, keep="new")
        return run
    if case == "queue":
        def run():
            worker = Worker("w", EngineConfig(size * 10, 1, 20000.0, 40.0, priority=True))
            worker.submit(Request("running", "running", "background", 16, 16, 1), 0.0)
            worker.schedule(0.0)
            for i in range(size):
                worker.submit(Request(str(i), str(i), "background", 16, 16, 1, tier=i % 2), 0.0)
                worker.schedule(0.0)
            return len(worker.queue)
        return run
    if case == "forecast":
        from atfm.board.forecaster import ExogenousModel, SessionForecaster
        from atfm.board.predictors import ProgressPredictor
        from atfm.board.state import SessionState
        from atfm.traces.synthetic import generate
        spec = WorkloadSpec(duration_s=600.0, seed=0, classes=[
            ClassSpec(cls="background", rate_per_hour=192.0, turns_mean=4, isl0=1000, isl_growth=100, osl_mean=40,
                      tools=[ToolSpec(name="pytest", weight=1.0, log_mu=np.log(30.0), log_sigma=0.5, signal="strong")]),
        ])
        train = generate(spec)
        model = SessionForecaster(ProgressPredictor().fit(train), ExogenousModel().fit(train), [30.0, 120.0, 300.0], n=128)
        states = [SessionState(session_id=str(i), cls="background", tenant="t", parent_session_id=None,
                               phase="tool_running", turn_index=0, tool_name="pytest", t_tool_start=0.0, ctx_tokens=1000,
                               progress=[{"t": 5.0, "completed": 10, "total": 100}]) for i in range(size)]
        return lambda: model.forecast(10.0, states, np.random.default_rng(7)).total("kv_blocks")
    if case == "duration":
        model = DurationModel()
        model._llm = {"background": np.random.default_rng(0).lognormal(1.0, 0.5, size)}
        def run():
            rng = np.random.default_rng(7)
            return np.stack([model.llm_duration_conditional("background", 1.0, 256, rng) for _ in range(256)])
        return run
    if case == "bootstrap":
        base = pd.DataFrame({"session_id": [f"s{i}" for i in range(size)], "value": np.random.default_rng(0).normal(size=size)})
        per = {"native": base, "a": base.assign(value=base.value + 1), "b": base.assign(value=base.value - 1)}
        return lambda: paired_bootstrap(per, lambda frame: frame.value.mean(), n_boot=100, rng=np.random.default_rng(7))
    if case == "h1":
        tools = [ToolSpec(name="pytest", weight=1.0, log_mu=np.log(30.0), log_sigma=0.5, signal="strong")]
        spec = WorkloadSpec(duration_s=600.0, seed=0, classes=[
            ClassSpec(cls="background", rate_per_hour=size * 6.0, turns_mean=4, isl0=1000, isl_growth=100, osl_mean=40, tools=tools),
        ])
        cfg = H1Config(name="h1", source="synthetic", synthetic=spec, tick_s=30.0,
                       horizons=[30.0, 120.0], n_samples=128, models=["B0", "M1", "M2"], out_dir=str(out))
        return lambda: run_h1(cfg)
    if case == "h2":
        cfg = H2SimConfig(name="h2", regime="short_tool", seeds=[0],
                         arms=["native", "proxy_rules", "forecast_M2_kv"], duration_s=300.0,
                         interactive_rate_per_hour=size * 6.0, background_rate_per_hour=size * 6.0,
                         engines=[{"kv_blocks": 8000, "max_batch": 8, "prefill_tps": 20000.0, "decode_tps": 40.0, "priority": True}],
                         window=8, out_dir=str(out))
        return lambda: run_h2sim(cfg)
    raise ValueError(case)


def fingerprint(value) -> str:
    if isinstance(value, pd.DataFrame):
        data = value.to_json(orient="split", double_precision=15).encode()
    elif isinstance(value, np.ndarray):
        data = value.tobytes()
    else:
        data = json.dumps(value, sort_keys=True).encode()
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="+", choices=["eviction", "queue", "forecast", "duration", "bootstrap", "h1", "h2"],
                        default=["eviction", "duration", "bootstrap"])
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true", help="Profile an extra run at the largest size")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 2 for size in args.sizes):
        parser.error("repeats must be positive and sizes must be at least 2")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in args.cases:
        for size in args.sizes:
            run = make_case(case, size, args.out / f"{case}_{size}")
            expected = fingerprint(run())
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                result = run()
                times.append(time.perf_counter() - start)
                assert fingerprint(result) == expected, f"non-deterministic result: {case}, {size}"
            row = {"case": case, "size": size, "median_s": float(np.median(times)), "min_s": min(times),
                   "max_s": max(times), "repeats": args.repeats, "result_sha256": expected}
            rows.append(row)
            pd.DataFrame(rows).to_csv(args.out / "timings.csv", index=False)
            print(f"{case:10} {size:6}: {row['median_s']:.6f} s", flush=True)
            if args.profile and size == max(args.sizes):
                profile = cProfile.Profile()
                profile.runcall(run)
                profile.dump_stats(str(args.out / f"{case}.prof"))
                with (args.out / f"{case}-profile.txt").open("w") as stream:
                    pstats.Stats(profile, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(35)
    cpuinfo = Path("/proc/cpuinfo")
    cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")), platform.processor()) if cpuinfo.exists() else platform.processor()
    sources = [*Path("src/atfm").rglob("*.py"), *Path("experiments/src/atfm_experiments").rglob("*.py")]
    meta = {"cpu": cpu, "logical_cpus": os.cpu_count(),
            "thread_env": {key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
            "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(sources)},
            "python": platform.python_version(), "platform": platform.platform(), "numpy": np.__version__,
            "pandas": pd.__version__, "cases": args.cases, "sizes": args.sizes,
            "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_status": subprocess.check_output(["git", "status", "--short"], text=True),
            "clock": "time.perf_counter", "setup": "excluded; one untimed warmup per case/size"}
    (args.out / "environment.json").write_text(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
