"""LMCache actuator: execute placement decisions through LMCache's controller instead of keep-alive touches.

LMCache's controller exposes `pin`, `move`, `lookup` and `clear` over HTTP, keyed by (instance, storage
location, token ids). Our directives are keyed by session; `PromptTokens` turns a session into its prefix
token ids (the proxy keeps the last prompt, a tokenizer is injected). Everything is fail-open: a dead
controller is counted, never raised. Endpoint shapes follow the documented controller API; verify them
against the multi-process controller before a hardware study."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Callable

from .directives import TierDirective, TouchDirective


@dataclass
class LMCacheConfig:
    url: str
    instance_id: str
    gpu_location: str = "LocalGPUBackend"
    cpu_location: str = "LocalCPUBackend"
    disk_location: str = "LocalDiskBackend"
    tier_map: dict[str, str] = field(default_factory=dict)     # extra tier name -> LMCache location
    timeout_s: float = 2.0
    paths: dict[str, str] = field(default_factory=lambda: {"pin": "/pin", "unpin": "/unpin", "move": "/move", "lookup": "/lookup",
                                                           "clear": "/clear", "check": "/check_finish", "health": "/health"})
    release_op: str = "unpin"       # how an expired pin is released: "unpin" (controller op) or "clear" (drop the entry)

    def location(self, tier: str) -> str:
        return {"gpu": self.gpu_location, "cpu": self.cpu_location, "disk": self.disk_location, **self.tier_map}.get(tier, tier)


class PromptTokens:
    """Session -> prefix token ids, cached until the session's prompt changes."""

    def __init__(self, tokenize: Callable[[list[dict]], list[int]], prompt_source: Callable[[str], list[dict] | None]):
        self.tokenize, self.prompt_source = tokenize, prompt_source
        self._cache: dict[str, tuple[str, list[int]]] = {}

    def get(self, session_id: str) -> list[int] | None:
        try:
            messages = self.prompt_source(session_id)
        except Exception:
            messages = None
        if not messages:
            return None
        key = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
        hit = self._cache.get(session_id)
        if hit is not None and hit[0] == key:
            return hit[1]
        toks = list(self.tokenize(messages))
        self._cache[session_id] = (key, toks)
        return toks


class LMCacheActuator:
    def __init__(self, cfg: LMCacheConfig, client=None, tokens: PromptTokens | None = None):
        if client is None:
            import httpx
            client = httpx.Client(timeout=cfg.timeout_s)
        self.cfg, self.client, self.tokens = cfg, client, tokens
        self.pinned: dict[str, tuple[float, str]] = {}      # session -> (expires_at, location)
        self.where: dict[str, str] = {}                      # session -> location we last placed it in
        self.errors = 0
        self.ops = {"pin": 0, "unpin": 0, "move": 0, "clear": 0, "lookup": 0}

    # ---- HTTP
    def _post(self, op: str, body: dict) -> dict | None:
        try:
            r = self.client.post(self.cfg.url.rstrip("/") + self.cfg.paths[op], json=body)
            if r.status_code != 200:
                self.errors += 1
                return None
            self.ops[op] = self.ops.get(op, 0) + 1
            return r.json()
        except Exception:
            self.errors += 1
            return None

    def _toks(self, session_id: str) -> list[int] | None:
        return self.tokens.get(session_id) if self.tokens is not None else None

    # ---- operations
    def pin(self, session_id: str, location: str, expires_at: float) -> dict:
        toks = self._toks(session_id)
        if not toks:
            return {"ok": False, "reason": "no prompt"}
        res = self._post("pin", {"instance_id": self.cfg.instance_id, "location": location, "tokens": toks})
        if res is None:
            return {"ok": False, "reason": "controller"}
        self.pinned[session_id] = (expires_at, location)
        self.where[session_id] = location
        return {"ok": True, "event_id": res.get("event_id"), "num_tokens": res.get("num_tokens", len(toks))}

    def move(self, session_id: str, dst_location: str) -> dict:
        toks = self._toks(session_id)
        if not toks:
            return {"ok": False, "reason": "no prompt"}
        src = self.where.get(session_id, self.cfg.gpu_location)
        res = self._post("move", {"src": {"instance_id": self.cfg.instance_id, "location": src},
                                  "dst": {"instance_id": self.cfg.instance_id, "location": dst_location}, "tokens": toks})
        if res is None:
            return {"ok": False, "reason": "controller"}
        self.where[session_id] = dst_location
        return {"ok": True, "event_id": res.get("event_id")}

    def lookup(self, session_id: str) -> list[dict]:
        toks = self._toks(session_id)
        if not toks:
            return []
        res = self._post("lookup", {"tokens": toks})
        if not res:
            return []
        if isinstance(res.get("res"), list):                       # list form
            return list(res["res"])
        out = []                                                   # documented form: {"event_id": ..., "<instance>": [location, hit]}
        for k, v in res.items():
            if k == "event_id" or not isinstance(v, (list, tuple)) or len(v) < 2:
                continue
            out.append({"instance_id": k, "location": v[0], "hit_tokens": int(v[1])})
        return out

    def release_expired(self, now: float) -> list[str]:
        """Unpin sessions whose directive expired (clear the pin in its location); returns the session ids."""
        out = []
        for sid, (exp, loc) in list(self.pinned.items()):
            if now >= exp:
                toks = self._toks(sid) or []
                self._post(self.cfg.release_op, {"instance_id": self.cfg.instance_id, "location": loc, "tokens": toks})
                del self.pinned[sid]
                out.append(sid)
        return out

    # ---- directives
    def apply_touch(self, d: TouchDirective, now: float) -> dict:
        """A keep-alive touch becomes a pin in the GPU-side location until the directive expires."""
        if d.expired(now):
            return {"ok": False, "reason": "expired"}
        return self.pin(d.session_id, self.cfg.gpu_location, d.expires_at)

    def apply_tier(self, d: TierDirective, now: float) -> dict:
        if d.expired(now):
            return {"ok": False, "reason": "expired"}
        loc = self.cfg.location(d.tier)
        if d.action in ("pin", "prefetch"):
            return self.pin(d.session_id, loc, d.expires_at)
        return self.move(d.session_id, loc)          # demote / promote
