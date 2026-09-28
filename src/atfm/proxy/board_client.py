"""Synchronous board client; deadline-aware calls use the proxy's remaining wait budget."""
from __future__ import annotations

import math
import time

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
        return self._validated(result)

    @staticmethod
    def _validated(result):
        service, tool = float(result["e_service_s"]), float(result["e_tool_next_s"])
        if (result.get("over_budget", False) or not math.isfinite(service)
                or not math.isfinite(tool) or service < 0 or tool < 0):
            raise ValueError("unavailable board prediction")
        return service, tool

    def expected_times_until(self, session_id: str, isl: int, osl: int, *, deadline: float):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('prediction expired before HTTP dispatch')
        try:
            response = self.client.post(f"{self.url}/predict", timeout=remaining,
                                        json={"session_id": session_id, "isl": isl, "osl": osl})
        except httpx.TimeoutException as exc:
            raise TimeoutError("prediction transport timed out") from exc
        response.raise_for_status()
        if time.monotonic() >= deadline:
            raise TimeoutError('prediction HTTP response exceeded deadline')
        return self._validated(response.json())

    def expected_tool_next(self, session_id: str) -> float:
        return float(self._predict(session_id, 0, 0).get("e_tool_next_s", 0.0))

    def expected_service(self, session_id: str, isl: int, osl: int) -> float:
        return float(self._predict(session_id, isl, osl).get("e_service_s", 0.0))
