"""Launch a CPU-only Dynamo stack (Mocker workers + frontend) with file discovery, for laptops."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


class LocalDynamo:
    def __init__(self, model: str = "Qwen/Qwen3-0.6B", workers: int = 2, port: int = 8000, blocks: int = 4096,
                 speedup: float = 10.0, log_dir: str = "runs/dynamo"):
        self.model, self.workers, self.port, self.blocks, self.speedup = model, workers, port, blocks, speedup
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.procs: list[subprocess.Popen] = []
        self.env_extra: dict[str, str] = {}

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _spawn(self, name: str, args: list[str]) -> subprocess.Popen:
        log = open(self.log_dir / f"{name}.log", "a")
        p = subprocess.Popen([sys.executable, "-m", *args], stdout=log, stderr=subprocess.STDOUT,
                             start_new_session=True, env={**os.environ, "DYN_HTTP_PORT": str(self.port), **self.env_extra})
        self.procs.append(p)
        return p

    def start(self, timeout_s: float = 120.0) -> None:
        self._spawn("mocker", ["dynamo.mocker", "--model-path", self.model, "--discovery-backend", "file",
                               "--num-workers", str(self.workers), "--num-gpu-blocks-override", str(self.blocks),
                               "--speedup-ratio", str(self.speedup)])
        self._spawn("frontend", ["dynamo.frontend", "--discovery-backend", "file", "--http-port", str(self.port),
                                 "--router-mode", "kv"])
        self._wait_ready(timeout_s)

    def _wait_ready(self, timeout_s):
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            if any(p.poll() is not None for p in self.procs):
                self.stop()
                raise RuntimeError(f"a dynamo process exited early; see {self.log_dir}")
            try:
                with urllib.request.urlopen(f"{self.base_url}/v1/models", timeout=2) as r:
                    models = json.loads(r.read().decode())
                if any(self.model in str(m) for m in models.get("data", [])):
                    return
            except Exception:
                pass
            time.sleep(2.0)
        self.stop()
        raise TimeoutError(f"dynamo frontend did not list {self.model} within {timeout_s}s; see {self.log_dir}")

    def stop(self) -> None:
        for p in self.procs:
            try:
                os.killpg(p.pid, signal.SIGTERM)
            except Exception:
                pass
        deadline = time.time() + 10
        for p in self.procs:
            while p.poll() is None and time.time() < deadline:
                time.sleep(0.2)
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except Exception:
                    pass
        self.procs = []

    def chat(self, messages: list[dict], max_tokens: int = 8, hints: dict | None = None,
             session_id: str | None = None) -> dict:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens}
        if hints:
            body["nvext"] = {"agent_hints": hints}
        headers = {"content-type": "application/json"}
        if session_id:
            headers["x-dynamo-session-id"] = session_id
        req = urllib.request.Request(f"{self.base_url}/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers=headers)
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *a):
        self.stop()
