"""The controllers' runtime (deploy): every interval, tick the board, fetch its directives, push holds and
touches to the proxy, and log tier and replica directives (advisory in v1). Fail-open throughout: a dead
board or proxy is counted, never raised (spec 10)."""
from __future__ import annotations

import json
import time
from pathlib import Path


class ControlLoop:
    def __init__(self, board_url: str, proxy_url: str, client=None, interval_s: float = 5.0, log_path: str | Path | None = None,
                 timeout_s: float = 2.0):
        if client is None:
            import httpx
            client = httpx.Client(timeout=timeout_s)
        self.board_url, self.proxy_url, self.client, self.interval_s = board_url.rstrip("/"), proxy_url.rstrip("/"), client, interval_s
        self.log_path = Path(log_path) if log_path else None
        self.totals = {"holds": 0, "touches": 0, "touch_tokens": 0, "errors": 0}

    def _log(self, rec: dict) -> None:
        if self.log_path is None:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")

    def step(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        s = {"sessions": 0, "holds": 0, "touches": 0, "touch_tokens": 0, "tier": 0, "replica": None, "errors": 0}
        try:
            r = self.client.post(f"{self.board_url}/tick")
            s["sessions"] = int(r.json().get("sessions", 0)) if r.status_code == 200 else 0
            d = self.client.get(f"{self.board_url}/directives").json()
        except Exception:
            s["errors"] += 1
            self.totals["errors"] += 1
            self._log({"t": now, "error": "board unreachable"})
            return s
        for h in d.get("holds", []):
            if h.get("expires_at") is not None and now >= float(h["expires_at"]):
                continue
            try:
                self.client.post(f"{self.proxy_url}/directives", json={"session_id": h["session_id"], "release_not_before": float(h["release_not_before"]),
                                                                      "reason": h.get("reason", "gdp"), "expires_at": h.get("expires_at")})
                s["holds"] += 1
            except Exception:
                s["errors"] += 1
        for t in d.get("touches", []):
            if t.get("expires_at") is not None and now >= float(t["expires_at"]):
                continue
            try:
                r = self.client.post(f"{self.proxy_url}/touch", json={"session_id": t["session_id"]})
                if r.status_code == 200 and r.json().get("ok"):
                    s["touches"] += 1
                    s["touch_tokens"] += int(r.json().get("prompt_tokens", 0))
            except Exception:
                s["errors"] += 1
        s["tier"] = len(d.get("tier", []))
        rep = d.get("replica")
        s["replica"] = int(rep["replicas_at_least"]) if rep else None
        for k in ("holds", "touches", "touch_tokens", "errors"):
            self.totals[k] += s[k]
        self._log({"t": now, **s, "tier": d.get("tier", []), "replica": rep})
        return s

    def run(self, steps: int | None = None) -> None:
        n = 0
        while steps is None or n < steps:
            self.step()
            n += 1
            time.sleep(self.interval_s)
