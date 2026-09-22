import numpy as np
from atfm.traces.synthetic import WorkloadSpec, ClassSpec, ToolSpec, Perturbation, generate, tail_share

def _spec(seed=0, long_weight=0.1, perturb=False):
    tools = [ToolSpec(name="bash", weight=1 - long_weight, log_mu=np.log(3.0), log_sigma=0.5, signal="none", backend_id="local"),
             ToolSpec(name="pytest", weight=long_weight, log_mu=np.log(300.0), log_sigma=0.6, signal="strong", backend_id="ci", spawn_prob=0.1)]
    perts = [Perturbation(backend_id="ci", t_start=1000.0, t_end=2000.0, factor=2.0)] if perturb else []
    return WorkloadSpec(duration_s=3600.0, seed=seed, perturbations=perts, classes=[
        ClassSpec(cls="background", rate_per_hour=60.0, turns_mean=8, isl0=2000, isl_growth=500, osl_mean=200, tools=tools),
        ClassSpec(cls="interactive", rate_per_hour=30.0, turns_mean=5, isl0=4000, isl_growth=800, osl_mean=150, tools=tools[:1],
                  think_log_mu=np.log(20.0), think_log_sigma=0.8)])

def test_generate_deterministic_and_valid():
    a = generate(_spec()); b = generate(_spec())
    assert a.df.equals(b.df) and len(a) > 100
    df = a.df
    tools = df[df["tool_name"].notna() & (df["tool_name"] != "__think__")]
    assert (tools["t_tool_end"] > tools["t_tool_start"]).all() and (tools["t_tool_start"] >= tools["t_last_token"] - 1e-9).all()
    assert set(df["class"]) == {"interactive", "background"} and df["source"].iloc[0] == "synthetic"
    strong = tools[tools["tool_name"] == "pytest"]
    assert all(len(ev) >= 1 for ev in strong["progress_events"]) and any(len(ev) > 3 for ev in strong["progress_events"])
    assert df["parent_session_id"].notna().any()
    assert (df[df["tool_name"] == "__think__"]["class"] == "interactive").all()

def test_tail_share_knob():
    lo = tail_share(generate(_spec(long_weight=0.02)))[1]
    hi = tail_share(generate(_spec(long_weight=0.3)))[1]
    assert hi > lo

def test_perturbation_slows_backend():
    df = generate(_spec(perturb=True)).df
    pert = df[df["perturbation_flag"]]
    assert len(pert) > 0 and (pert["backend_id"] == "ci").all()
    dur = lambda d: (d["t_tool_end"] - d["t_tool_start"])
    ci = df[(df["tool_name"] == "pytest")]
    assert dur(ci[ci["perturbation_flag"]]).median() > dur(ci[~ci["perturbation_flag"]]).median()
