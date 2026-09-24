from __future__ import annotations

import json
import urllib.request


def gate_allowed_at(url: str | None, session_id: str, kind: str, *, timeout_s: float = 0.2) -> float | None:
    """Ask the proxy's launch gate. Fail-open: any error means 'allowed now' (None)."""
    if not url:
        return None
    try:
        req = urllib.request.Request(
            f"{url.rstrip('/')}/gate",
            data=json.dumps({"session_id": session_id, "kind": kind}).encode(),
            headers={"content-type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            body = json.loads(r.read().decode())
        v = body.get("allowed_at")
        return None if v is None else float(v)
    except Exception:
        return None
