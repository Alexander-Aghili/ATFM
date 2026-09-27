"""Six-arm closed-loop serving experiment on synthetic regimes (spec section 9), paired by seed."""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from atfm.eval.serving import paired_bootstrap, paired_contrasts, serving_metrics
from atfm.proxy.config import ProxyConfig
from atfm.sim.core import Simulator
from atfm.sim.engine import EngineConfig
from atfm.sim.forecast_arm import ForecastPolicy, GdpLite, OracleRuleNoIdxPolicy, OracleRulePolicy, fit_predictor_on_programs
from atfm.sim.kv_placement import ForecastKvPolicy, ForecastTouchPolicy, OracleKvPolicy, OracleTouchPolicy
from atfm.sim.programs import programs_from_table
from atfm.schema.trace import TraceTable
import atfm
from atfm.sim.policies import NativePolicy, OraclePolicy, ProxyRulesPolicy, WorkingSetPolicy
from atfm.sim.programs import programs_from_spec
from atfm.traces.synthetic import ClassSpec, ToolSpec, WorkloadSpec

ARMS = ["native", "proxy_rules", "forecast_M1", "forecast_M2", "oracle", "oracle_rule", "working_set",
        "forecast_M1_kv", "forecast_M2_kv", "oracle_kv", "oracle_rule_noidx",
        "forecast_M1_kv_size", "forecast_M2_kv_size", "oracle_kv_size",
        "forecast_M1_kv_cw", "forecast_M2_kv_cw", "oracle_kv_cw",
        "forecast_M1_touch", "forecast_M2_touch", "oracle_touch"]


def default_contrasts(arms: list[str]) -> list[tuple[str, str]]:
    """Direct paired contrasts worth reporting when both arms ran: M2 against M1, forecast placement
    against true-return-time placement, size-aware against size-blind."""
    cands = [("forecast_M2", "forecast_M1"), ("forecast_M2_kv", "forecast_M1_kv"),
             ("forecast_M2_kv", "oracle_kv"), ("forecast_M1_kv", "oracle_kv"),
             ("forecast_M2_kv_size", "forecast_M2_kv"), ("forecast_M1_kv_size", "forecast_M1_kv"),
             ("oracle_kv_size", "oracle_kv"), ("forecast_M2_kv_size", "oracle_kv_size"),
             ("forecast_M2_kv_cw", "forecast_M2_kv"), ("forecast_M1_kv_cw", "forecast_M1_kv"),
             ("oracle_kv_cw", "oracle_kv"), ("forecast_M2_kv_cw", "oracle_kv_cw"), ("forecast_M2_kv_cw", "forecast_M1_kv_cw")]
    return [(a, b) for a, b in cands if a in arms and b in arms]
ABLATIONS = ["forecast_M1_nohold", "forecast_M2_nohold"]


class H2SimConfig(BaseModel):
    name: str
    regime: Literal["short_tool", "long_tool", "interactive_long_tool", "trace"]
    trace_path: str | None = None            # regime "trace": parquet trace table replayed as programs
    trace_rate_per_hour: float | None = None # overlay rate for the replay (None = as recorded)
    seeds: list[int] = Field(default_factory=lambda: [0, 1, 2])
    arms: list[str] = Field(default_factory=lambda: list(ARMS))
    engines: list[dict] = Field(default_factory=lambda: [{"kv_blocks": 8000, "max_batch": 8, "prefill_tps": 20000.0, "decode_tps": 40.0, "priority": True}])
    window: int = 8
    beta: float = 0.5
    slo_ttft_s: float = 2.0
    working_set_budget: int = 6000
    duration_s: float = 3600.0
    interactive_rate_per_hour: float = 120.0
    background_rate_per_hour: float = 240.0
    train_seed_offset: int = 1000
    max_hold_s: float = 600.0        # cap on any policy hold, enforced in the simulator core
    contrasts: list[list[str]] = Field(default_factory=list)   # explicit [a, b] pairs; empty = default_contrasts(arms)
    kv_bg_weight: float = 3.0        # class weight on background absence for the *_cw placement arms
    touch_budget_per_s: float = 1.0  # keep-alive touches per second for the *_touch arms
    touch_horizon_s: float = 30.0
    touch_age_s: float = 10.0
    out_dir: str = "runs"


def regime_spec(regime: str, duration_s: float = 3600.0, seed: int = 0, it_rate: float = 120.0, bg_rate: float = 240.0) -> WorkloadSpec:
    if regime == "short_tool":
        tools = [ToolSpec(name="bash", weight=0.85, log_mu=np.log(2.0), log_sigma=0.7, signal="none"),
                 ToolSpec(name="pytest", weight=0.15, log_mu=np.log(20.0), log_sigma=0.6, signal="strong", backend_id="ci")]
    else:
        tools = [ToolSpec(name="bash", weight=0.5, log_mu=np.log(3.0), log_sigma=0.6, signal="none"),
                 ToolSpec(name="pytest", weight=0.3, log_mu=np.log(180.0), log_sigma=0.6, signal="strong", backend_id="ci", spawn_prob=0.05),
                 ToolSpec(name="build", weight=0.2, log_mu=np.log(400.0), log_sigma=0.5, signal="weak", backend_id="ci")]
    if regime == "interactive_long_tool":
        # interactive sessions also run long, signalled tools (test suites, builds), so *when* they come back
        # is what the hold rule has to predict and progress-aware prediction (M2) can differ from M1
        it_tools = [ToolSpec(name="bash", weight=0.5, log_mu=np.log(3.0), log_sigma=0.6, signal="none"),
                    ToolSpec(name="pytest", weight=0.35, log_mu=np.log(120.0), log_sigma=0.6, signal="strong", backend_id="ci"),
                    ToolSpec(name="build", weight=0.15, log_mu=np.log(300.0), log_sigma=0.5, signal="weak", backend_id="ci")]
    else:
        it_tools = [tools[0]]
    return WorkloadSpec(duration_s=duration_s, seed=seed, classes=[
        ClassSpec(cls="background", rate_per_hour=bg_rate, turns_mean=10, isl0=3000, isl_growth=600, osl_mean=200, tools=tools, deadline_s=1800.0),
        ClassSpec(cls="interactive", rate_per_hour=it_rate, turns_mean=6, isl0=4000, isl_growth=800, osl_mean=150,
                  tools=it_tools, think_log_mu=np.log(20.0), think_log_sigma=0.8)])


def _arm(name: str, cfg: H2SimConfig, engines: list[EngineConfig], train_programs, rng):
    pcfg = ProxyConfig(upstream_url="sim", beta=cfg.beta, prefill_tps=engines[0].prefill_tps, decode_tps=engines[0].decode_tps)
    if name == "native":
        return NativePolicy(priority_by_class=True)
    if name == "proxy_rules":
        return ProxyRulesPolicy(cfg.window, pcfg)
    if name == "oracle":
        return OraclePolicy(cfg.window, pcfg, hold=True)
    if name == "oracle_touch":
        return OracleTouchPolicy(cfg.window, pcfg, horizon_s=cfg.touch_horizon_s, age_s=cfg.touch_age_s, budget_per_s=cfg.touch_budget_per_s)
    if name in ("forecast_M1_touch", "forecast_M2_touch"):
        pred, table = fit_predictor_on_programs(name.split("_")[1], train_programs, engines, rng)
        return ForecastTouchPolicy(cfg.window, pcfg, pred, table, horizons=[30.0, 120.0, 300.0], n=64,
                                   horizon_s=cfg.touch_horizon_s, age_s=cfg.touch_age_s, budget_per_s=cfg.touch_budget_per_s)
    if name in ("oracle_kv", "oracle_kv_size", "oracle_kv_cw"):
        return OracleKvPolicy(cfg.window, pcfg, size_aware=name.endswith("_size"),
                              bg_weight=cfg.kv_bg_weight if name.endswith("_cw") else 1.0)
    if name in ("forecast_M1_kv", "forecast_M2_kv", "forecast_M1_kv_size", "forecast_M2_kv_size",
                "forecast_M1_kv_cw", "forecast_M2_kv_cw"):
        pred, table = fit_predictor_on_programs(name.split("_")[1], train_programs, engines, rng)
        return ForecastKvPolicy(cfg.window, pcfg, pred, table, horizons=[30.0, 120.0, 300.0], n=64,
                                size_aware=name.endswith("_size"), bg_weight=cfg.kv_bg_weight if name.endswith("_cw") else 1.0)
    if name == "oracle_rule_noidx":
        return OracleRuleNoIdxPolicy(cfg.window, pcfg, horizons=[30.0, 120.0, 300.0], gdp=GdpLite(max_hold_s=cfg.max_hold_s))
    if name == "oracle_rule":
        return OracleRulePolicy(cfg.window, pcfg, horizons=[30.0, 120.0, 300.0], gdp=GdpLite(max_hold_s=cfg.max_hold_s))
    if name == "working_set":
        return WorkingSetPolicy(cfg.working_set_budget)
    if name in ARMS[2:4] or name in ABLATIONS:
        kind = name.split("_")[1]
        pred, table = fit_predictor_on_programs(kind, train_programs, engines, rng)
        pol = ForecastPolicy(cfg.window, pcfg, pred, table, horizons=[30.0, 120.0, 300.0], n=128, hold=not name.endswith("_nohold"),
                             gdp=GdpLite(max_hold_s=cfg.max_hold_s))
        pol.name = name
        return pol
    raise ValueError(f"unknown arm {name}; choose from {ARMS + ABLATIONS}")


def _sha256_file(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_manifest(out: Path, cfg: H2SimConfig, inputs: list[dict]) -> dict:
    """Provenance for a run directory (spec 9): git SHA, config hash, inputs with hashes, package version."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"
    m = {"git_sha": sha, "config_sha256": hashlib.sha256(cfg.model_dump_json().encode()).hexdigest(),
         "atfm_version": atfm.__version__, "inputs": inputs, "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.json").write_text(json.dumps(m, indent=2))
    return m


def split_table_for_training(table: TraceTable, test_fraction: float = 0.5, seed: int = 0) -> tuple[TraceTable, TraceTable]:
    """Hold out whole session families (a child follows its root) by a stable hash, so predictors are never
    fit on the sessions they are scored against (spec 9)."""
    df = table.df
    roots = df["parent_session_id"].where(df["parent_session_id"].notna(), df["session_id"]).astype(str)
    frac = roots.map(lambda r: int(hashlib.sha256(f"{seed}:{r}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF)
    test = frac < test_fraction
    return TraceTable(df[~test].reset_index(drop=True)), TraceTable(df[test].reset_index(drop=True))


_TRACE_SPLIT = {"test_fraction": 0.5, "seed": 0}


def _programs_for(cfg: H2SimConfig, seed: int, train: bool):
    """Programs for one seed: synthetic from the regime spec, or a replayed trace table (overlaid at
    `trace_rate_per_hour`) split by session family into a training half and an evaluation half."""
    rng_seed = seed + cfg.train_seed_offset if train else seed
    if cfg.regime == "trace":
        table = TraceTable.from_parquet(cfg.trace_path)
        train_t, test_t = split_table_for_training(table, **_TRACE_SPLIT)
        return programs_from_table(train_t if train else test_t, cfg.trace_rate_per_hour, cfg.duration_s, np.random.default_rng(rng_seed))
    spec = regime_spec(cfg.regime, cfg.duration_s, rng_seed if train else seed, cfg.interactive_rate_per_hour, cfg.background_rate_per_hour)
    return programs_from_spec(spec, np.random.default_rng(rng_seed))


def build_simulator(cfg: H2SimConfig, programs, engines: list[EngineConfig], policy, seed: int) -> Simulator:
    return Simulator(programs, engines, policy, slo_ttft_s=cfg.slo_ttft_s, max_hold_s=cfg.max_hold_s,
                     rng=np.random.default_rng(seed + 13))


def run_h2sim(cfg: H2SimConfig) -> pd.DataFrame:
    out = Path(cfg.out_dir) / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    engines = [EngineConfig(**e) for e in cfg.engines]
    rows, per_arm_sessions = [], {a: [] for a in cfg.arms}
    write_manifest(out, cfg, inputs=[{"path": cfg.trace_path, "sha256": _sha256_file(cfg.trace_path), "split": _TRACE_SPLIT}] if cfg.trace_path else [])
    for seed in cfg.seeds:
        train_programs = _programs_for(cfg, seed, train=True)
        base_arrivals = None
        for arm in cfg.arms:
            programs = _programs_for(cfg, seed, train=False)  # identical programs for every arm
            arrivals = [p.t_arrival for p in programs]
            if base_arrivals is None:
                base_arrivals = arrivals
            assert arrivals == base_arrivals, "paired design broken: arrivals differ across arms"
            policy = _arm(arm, cfg, engines, train_programs, np.random.default_rng(seed + 7))
            sim = build_simulator(cfg, programs, engines, policy, seed)
            log = sim.run()
            log.to_parquet(out / f"log_{arm}_{seed}.parquet", index=False)
            m = serving_metrics(log, sim.session_log, cfg.slo_ttft_s, cfg.duration_s, len(engines), makespan_s=sim.now)
            m["caps"] = sim.caps
            m["touches"], m["touch_hits"], m["touch_misses"], m["touch_fails"] = sim.touches, sim.touch_hits, sim.touch_misses, sim.touch_fails
            m["touch_prefill_tokens"] = sim.touch_prefill_tokens
            m["max_imposed_delay_by_tenant"] = json.dumps(m["max_imposed_delay_by_tenant"])
            rows.append({"arm": arm, "seed": seed, **m})
            sess = pd.DataFrame(sim.session_log)
            sess["session_id"] = sess["session_id"] + f"@{seed}"
            it = log[(log["class"] == "interactive") & (log["turn_index"] > 0)].copy()
            it["ttft"] = it["t_first_token"] - it["t_arrival"]
            slo = it.groupby("session_id")["ttft"].apply(lambda s: float((s <= cfg.slo_ttft_s).mean())).rename("slo").reset_index()  # session-weighted
            slo["session_id"] = slo["session_id"] + f"@{seed}"
            per_arm_sessions[arm].append(sess.merge(slo, on="session_id", how="left"))
    df = pd.DataFrame(rows)
    df.to_csv(out / "metrics.csv", index=False)
    per = {a: pd.concat(v, ignore_index=True) for a, v in per_arm_sessions.items()}
    metric_fns = {
        "slo_attainment_sessions": lambda d: float(d["slo"].dropna().mean()) if d["slo"].notna().any() else float("nan"),
        "bg_jct_mean": lambda d: float((d.loc[d["class"] == "background", "t_end"] - d.loc[d["class"] == "background", "t_start"]).mean()),
        "deadline_hit_rate": lambda d: float(1.0 - d.loc[d["deadline"].notna(), "missed"].mean()),
    }
    paired = []
    for metric, fn in metric_fns.items():
        pb = paired_bootstrap(per, fn, n_boot=300, rng=np.random.default_rng(0))
        pb["metric"] = metric
        paired.append(pb)
    pd.concat(paired, ignore_index=True).to_csv(out / "paired.csv", index=False)
    pairs = [tuple(p) for p in cfg.contrasts] or default_contrasts(cfg.arms)
    if pairs:
        contrasts = []
        for metric, fn in metric_fns.items():
            c = paired_contrasts(per, fn, pairs, n_boot=300, rng=np.random.default_rng(0))
            c["metric"] = metric
            contrasts.append(c)
        pd.concat(contrasts, ignore_index=True).to_csv(out / "contrasts.csv", index=False)
    df[["arm", "seed", "queue_proxy_share", "queue_worker_share", "mean_held_s_background", "hold_kv_block_s",
        "evictions_caused_by_holds"]].to_csv(out / "queue_location.csv", index=False)
    (out / "config.json").write_text(cfg.model_dump_json(indent=2))
    return df
