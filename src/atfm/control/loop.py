"""The controllers' runtime (deploy): every interval, tick the board, fetch its directives, push holds and
touches to the proxy, and log tier and replica directives (advisory in v1). Fail-open throughout: a dead
board or proxy is counted, never raised (spec 10)."""
from __future__ import annotations

import json
import time
from pathlib import Path

from atfm.control.directives import HoldBatch, HoldUpdate, MAX_HOLD_BATCH, TierDirective, TouchDirective


class ControlLoop:
    def __init__(self, board_url: str, proxy_url: str, client=None, interval_s: float = 5.0, log_path: str | Path | None = None,
                 timeout_s: float = 2.0, lmcache=None, hold_batch_size: int = MAX_HOLD_BATCH):
        """With an `LMCacheActuator`, touches become pins and tier directives are executed (pin / move);
        without one, touches go to the proxy's /touch and tier directives are only logged."""
        if not 1 <= hold_batch_size <= MAX_HOLD_BATCH:
            raise ValueError(f"hold_batch_size must be in [1, {MAX_HOLD_BATCH}]")
        self.hold_batch_size = hold_batch_size
        if client is None:
            import httpx
            client = httpx.Client(timeout=timeout_s)
        self.board_url, self.proxy_url, self.client, self.interval_s = board_url.rstrip("/"), proxy_url.rstrip("/"), client, interval_s
        self.lmcache = lmcache
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
        s = {"sessions": 0, "holds": 0, "touches": 0, "touch_tokens": 0, "tier": 0, "replica": None, "errors": 0, "pins": 0, "tier_applied": 0}
        try:
            r = self.client.post(f"{self.board_url}/tick")
            s["sessions"] = int(r.json().get("sessions", 0)) if r.status_code == 200 else 0
            d = self.client.post(f"{self.board_url}/directives").json()
        except Exception:
            s["errors"] += 1
            self.totals["errors"] += 1
            self._log({"t": now, "error": "board unreachable"})
            return s
        holds = self._valid_holds(d, now, s)
        self._send_holds(holds, s)
        self._apply_touches(d, now, s)
        self._apply_tiers(d, now, s)
        self._record_step(d, now, s)
        return s

    def _record_step(self, d, now, s):
        rep = d.get("replica")
        s["replica"] = int(rep["replicas_at_least"]) if rep else None
        for k in ("holds", "touches", "touch_tokens", "errors"):
            self.totals[k] += s[k]
        self._log({"t": now, **s, "tier": d.get("tier", []), "replica": rep})

    def _apply_tiers(self, d, now, s):
        s["tier"] = len(d.get("tier", []))
        if self.lmcache is not None:
            for t in d.get("tier", []):
                if self.lmcache.apply_tier(TierDirective(**{k: v for k, v in t.items() if k != "kind"}), now).get("ok"):
                    s["tier_applied"] += 1
            self.lmcache.release_expired(now)
            s["errors"] += self.lmcache.errors - getattr(self, "_lm_err", 0)
            self._lm_err = self.lmcache.errors

    def _apply_touches(self, d, now, s):
        for t in d.get("touches", []):
            if t.get("expires_at") is not None and now >= float(t["expires_at"]):
                continue
            if self.lmcache is not None:
                if self.lmcache.apply_touch(TouchDirective(**{k: v for k, v in t.items() if k != "kind"}), now).get("ok"):
                    s["pins"] += 1
                continue
            self._proxy_touch(t, s)

    def _proxy_touch(self, t, s):
        try:
            r = self.client.post(f"{self.proxy_url}/touch", json={"session_id": t["session_id"]})
            if r.status_code == 200 and r.json().get("ok"):
                s["touches"] += 1
                s["touch_tokens"] += int(r.json().get("prompt_tokens", 0))
        except Exception:
            s["errors"] += 1

    def _send_holds(self, holds, s):
        for start in range(0, len(holds), self.hold_batch_size):
            batch = HoldBatch(holds=holds[start:start + self.hold_batch_size])
            try:
                applied = self._apply_hold_batch(batch)
                s["holds"] += applied
            except Exception:
                s["errors"] += 1

    def _apply_hold_batch(self, batch):
        response = self.client.post(f"{self.proxy_url}/directives/batch", json=batch.model_dump())
        result = response.json()
        applied = result.get("applied")
        expired = result.get("expired")
        if (response.status_code != 200 or result.get("ok") is not True
                or type(applied) is not int or type(expired) is not int
                or min(applied, expired) < 0 or applied + expired != len(batch.holds)):
            raise ValueError("invalid hold-batch acknowledgement")
        return applied

    def _valid_holds(self, d, now, s):
        holds = []
        for h in d.get("holds", []):
            try:
                hold = HoldUpdate.model_validate(h)
            except ValueError:
                s["errors"] += 1
                continue
            if hold.expires_at is None or now < hold.expires_at:
                holds.append(hold)
        return holds

    def run(self, steps: int | None = None) -> None:
        n = 0
        while steps is None or n < steps:
            self.step()
            n += 1
            time.sleep(self.interval_s)
