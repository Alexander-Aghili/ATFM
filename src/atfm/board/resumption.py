"""Per-session summaries shared by serving and simulated placement controllers."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from atfm.board.predictors.base import SessionPredictor
from atfm.board.state import SessionState
from atfm.schema.forecast import ResumptionQuantiles


def resumption_quantiles(
    predictor: SessionPredictor,
    states: Iterable[SessionState],
    now: float,
    n: int,
    rng: np.random.Generator,
    *,
    skip_errors: bool = False,
) -> dict[str, ResumptionQuantiles]:
    """Return q10, q50 and q90 seconds until each session's next call.

    Quantiles are conditional on finite draws; sessions with no finite draws
    are omitted. This is a placement summary, not a probability of returning.
    Draw once per state, in input order, using the caller's random generator.

    Serving may opt into ``skip_errors`` to isolate a failed prediction to one
    session. Simulation defaults to propagating errors so model defects cannot
    silently change experiment results.
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
