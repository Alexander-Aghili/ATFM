"""The proxy's predictor when a board service is configured: synchronous HTTP calls to the board's
`/predict`, run by the proxy inside its thread pool under the 50 ms budget. Fail-open: any error returns
the proxy's defaults (0.0), which is what it would use without a board."""
from __future__ import annotations

import math

import httpx


class BoardClient:
    def __init__(self, board_url: str, timeout_s: float = 0.05):
        self.url = board_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)

    def _predict(self, session_id: str, isl: int, osl: int) -> dict:
        try:
            r = self.client.post(f"{self.url}/predict", json={"session_id": session_id, "isl": isl, "osl": osl})
            return r.json() if r.status_code == 200 else {}
        except Exception:
            return {}

    def expected_times(self, session_id: str, isl: int, osl: int) -> tuple[float, float]:
        """Fetch both estimates once; reject unavailable results for proxy fallback."""
        result = self._predict(session_id, isl, osl)
        service, tool = float(result["e_service_s"]), float(result["e_tool_next_s"])
        if (result.get("over_budget", False) or not math.isfinite(service)
                or not math.isfinite(tool) or service < 0 or tool < 0):
            raise ValueError("unavailable board prediction")
        return service, tool

    def expected_tool_next(self, session_id: str) -> float:
        return float(self._predict(session_id, 0, 0).get("e_tool_next_s", 0.0))

    def expected_service(self, session_id: str, isl: int, osl: int) -> float:
        return float(self._predict(session_id, isl, osl).get("e_service_s", 0.0))
