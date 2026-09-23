from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ForecastSnapshot:
    t: float
    horizons: list[float]
    model_id: str
    samples: dict[str, dict[str, np.ndarray]]  # [target][class] -> (H, n)
    endogenous_fraction: dict[str, np.ndarray] = field(default_factory=dict)  # [class] -> (H,)

    def quantiles(self, target: str, cls: str, q: float) -> np.ndarray:
        return np.quantile(self.samples[target][cls], q, axis=1)

    def total(self, target: str) -> np.ndarray:
        return sum(self.samples[target].values())
