"""Per-session summaries shared by serving and simulated placement controllers."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from atfm.board.predictors.base import SessionPredictor
from atfm.board.state import SessionState
from atfm.schema.forecast import ResumptionQuantiles


def resumption_quantiles(predictor: SessionPredictor, states: Iterable[SessionState], now: float,
                         n: int, rng: np.random.Generator, *, skip_errors: bool = False,
                         ) -> dict[str, ResumptionQuantiles]:
    """Conditional finite-draw quantiles, in state/RNG order; omit nonreturners.

    Serving can skip failed sessions; simulation propagates errors by default.
    These quantiles do not estimate the probability of returning.
    """
    result = {}
    for state in states:
        try:
            samples = predictor.resumption(state, now, n, rng)
            finite = samples[np.isfinite(samples)]
            if len(finite):
                q10, q50, q90 = np.quantile(finite, [0.1, 0.5, 0.9])
                result[state.session_id] = (float(q10), float(q50), float(q90))
        except Exception:
            if not skip_errors:
                raise
    return result
