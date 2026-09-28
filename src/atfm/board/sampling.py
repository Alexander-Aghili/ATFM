"""Sampling primitives shared by empirical predictors and fleet aggregation."""
from __future__ import annotations

import numpy as np


def empirical_draw(values: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform replacement draws from a one-dimensional empirical distribution.

    Integer indexing avoids Generator.choice's generic shape/axis machinery for
    small repeated draws. It consumes the same integer stream as unweighted
    choice(size=n, replace=True); parity tests cover supported NumPy generators.
    """
    return values[rng.integers(len(values), size=n)]
