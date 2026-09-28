from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


CLASSES = ("interactive", "background")
TARGETS = ("kv_blocks", "prefill_tokens")

ResumptionQuantiles = tuple[float, float, float]


@dataclass
class ForecastSnapshot:
    """Cumulative demand draws at each horizon, measured in seconds from ``t``.

    ``samples[target][class]`` has shape (horizons, draws); endogenous fractions
    have shape (horizons,) per class. Aggregating classes preserves draw indices.
    """

    t: float
    horizons: list[float]
    model_id: str
    samples: dict[str, dict[str, np.ndarray]]
    endogenous_fraction: dict[str, np.ndarray] = field(default_factory=dict)

    def quantiles(self, target: str, cls: str, q: float) -> np.ndarray:
        return np.quantile(self.samples[target][cls], q, axis=1)

    def total(self, target: str) -> np.ndarray:
        return sum(self.samples[target].values())
