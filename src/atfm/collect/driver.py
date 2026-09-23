"""Scripted trace collection: real tools through the sidecar, LLM calls through the proxy (or synthetic)."""
from __future__ import annotations

import json
import tempfile
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from atfm.bus import JsonlBus
from atfm.schema.events import LlmDone, LlmRequest, SessionStart
from atfm.sidecar.minisweagent import SidecarConfig

from .jobs import CollectionSpec, JobSpec


def no_llm(bus):
    """Synthetic zero-duration LLM calls so tool phases still pair with calls in the trace table."""
    turns: dict[str, int] = {}
    started: set[str] = set()

    def call(session_id, cls, tenant, deadline, prompt, max_tokens):
        t = time.time()
        if session_id not in started:
            started.add(session_id)
            bus.publish(SessionStart(t=t, session_id=session_id, tenant=tenant, cls=cls, deadline=deadline))
        k = turns.get(session_id, 0)
        turns[session_id] = k + 1
        rid = uuid.uuid4().hex[:12]
        bus.publish(LlmRequest(t=t, session_id=session_id, turn_index=k, request_id=rid, isl=max(1, len(prompt) // 4)))
        bus.publish(LlmDone(t=t + 1e-3, session_id=session_id, request_id=rid, osl=0))
        return {"synthetic": True}

    return call


def proxy_chat(proxy_url: str, model: str):
    def call(session_id, cls, tenant, deadline, prompt, max_tokens):
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
        headers = {"content-type": "application/json", "x-atfm-session": session_id, "x-atfm-class": cls,
                   "x-atfm-tenant": tenant}
        if deadline is not None:
            headers["x-atfm-deadline"] = str(deadline)
        req = urllib.request.Request(f"{proxy_url.rstrip('/')}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.loads(r.read().decode())

    return call


def _default_env_factory(job: JobSpec, cfg: SidecarConfig):
    from atfm.sidecar.minisweagent import SidecarDockerEnvironment, SidecarLocalEnvironment
    if job.image:
        return SidecarDockerEnvironment(sidecar=cfg, image=job.image, timeout=int(job.timeout_s), cwd="/w",
                                        run_args=["--rm"])
    return SidecarLocalEnvironment(sidecar=cfg, timeout=int(job.timeout_s), cwd=tempfile.mkdtemp(prefix="atfm-"))


def _expand_setup(item) -> str:
    if isinstance(item, str):
        return item
    src = Path(item["file"]).read_text()
    return f"mkdir -p {Path(item['to']).parent} && cat > {item['to']} <<'ATFM_EOF'\n{src}\nATFM_EOF"


def _run_session(job: JobSpec, k: int, spec: CollectionSpec, bus, env_factory, llm, clock) -> dict:
    sid = f"{job.name}-{k}-{uuid.uuid4().hex[:6]}"
    deadline = None if job.deadline_s is None else clock() + job.deadline_s
    cfg = SidecarConfig(session_id=sid, tenant=job.tenant, cls=job.cls, bus=bus, gate_url=spec.proxy_url, clock=clock)
    tools = errors = 0
    env = None
    try:
        env = env_factory(job, cfg)
        for s in job.setup:
            out = env.execute({"command": _expand_setup(s)})
            tools += 1
            if out["returncode"] != 0:
                errors += 1
        for turn in job.turns:
            llm(sid, job.cls, job.tenant, deadline, turn.prompt, turn.max_tokens)
            out = env.execute({"command": turn.cmd})
            tools += 1
            if out["returncode"] not in (0, 1):  # test failures (1) are legitimate outcomes
                errors += 1
        llm(sid, job.cls, job.tenant, deadline, "final", 8)
    except Exception:
        errors += 1
    finally:
        cleanup = getattr(env, "cleanup", None)
        if callable(cleanup):
            try:
                cleanup()
            except Exception:
                pass
    return {"tools": tools, "errors": errors}


def run_collection(spec: CollectionSpec, env_factory=None, llm=None, bus=None, clock=time.time) -> dict:
    bus = bus if bus is not None else JsonlBus(spec.events_path)
    env_factory = env_factory or _default_env_factory
    llm = llm or (proxy_chat(spec.proxy_url, spec.model) if spec.proxy_url else no_llm(bus))
    work = [(job, k) for job in spec.jobs for k in range(job.repeat)]
    totals = {"sessions": 0, "tools": 0, "errors": 0}
    with ThreadPoolExecutor(max_workers=max(1, spec.concurrency)) as pool:
        for r in pool.map(lambda jk: _run_session(jk[0], jk[1], spec, bus, env_factory, llm, clock), work):
            totals["sessions"] += 1
            totals["tools"] += r["tools"]
            totals["errors"] += r["errors"]
    return totals
