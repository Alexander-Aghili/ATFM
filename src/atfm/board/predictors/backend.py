from __future__ import annotations

import numpy as np

from .progress import ProgressPredictor


class BackendFactor:
    """Per-backend latent log-speed factor: EWMA mean and variance of observed/expected duration ratios."""

    def __init__(self, alpha: float = 0.3, sigma_floor: float = 0.05):
        self.alpha, self.sigma_floor = alpha, sigma_floor
        self.mu: dict[str, float] = {}
        self.var: dict[str, float] = {}

    def update(self, backend_id: str, observed_ratio: float) -> None:
        z = float(np.log(max(observed_ratio, 1e-6)))
        if backend_id not in self.mu:
            self.mu[backend_id] = z
            self.var[backend_id] = self.sigma_floor ** 2
            return
        mu = self.mu[backend_id]
        self.mu[backend_id] = (1 - self.alpha) * mu + self.alpha * z
        self.var[backend_id] = (1 - self.alpha) * self.var[backend_id] + self.alpha * (z - mu) ** 2

    def draw(self, backend_id: str | None, n: int, rng: np.random.Generator) -> np.ndarray:
        if backend_id not in self.mu:
            return np.ones(n)
        sigma = max(np.sqrt(self.var[backend_id]), self.sigma_floor)
        return np.exp(rng.normal(self.mu[backend_id], sigma, size=n))


class BackendPredictor(ProgressPredictor):
    """M3: M2 plus a shared latent speed factor per backend."""

    name = "M3_backend"

    def __init__(self):
        super().__init__()
        self.factor = BackendFactor()

    def observe_completion(self, backend_id: str | None, tool: str | None, actual_duration: float) -> None:
        if backend_id is None:
            return
        self.factor.update(backend_id, actual_duration / max(self.dm.mean(tool), 1e-6))

    def _scale(self, s, rem, n, rng, factors):
        if factors is not None and s.backend_id in factors:
            f = factors[s.backend_id]
        else:
            f = self.factor.draw(s.backend_id, n, rng)
        return rem * f
