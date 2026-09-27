from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from atfm.board.state import SessionState
from atfm.schema.trace import TraceTable


class SessionPredictor(ABC):
    """Per-session sampling interface used by fleet and placement forecasting.

    Sampling methods return one-dimensional arrays of length ``n`` and consume
    only the supplied random generator. Inputs describe information available
    at the forecast tick; fitting must not use held-out evaluation sessions.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, train: TraceTable) -> "SessionPredictor":
        """Learn from the training trace and return the fitted predictor."""

    @abstractmethod
    def resumption(self, s: SessionState, now: float, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw seconds from ``now`` to the next call; infinity denotes no return."""

    @abstractmethod
    def next_call_isl(self, s: SessionState, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw the next call's input length in tokens, including retained context."""

    @abstractmethod
    def spawn(self, s: SessionState, horizon: float, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw integer child-session counts within ``horizon`` seconds."""


class SeriesPredictor(ABC):
    """Aggregate time-series predictor (B0/B1): past demand values in, samples of the next out."""

    name: str = "series"

    @abstractmethod
    def predict(self, history: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``n`` demand values from fully observed history for one series.

        History is ordered oldest to newest and may be empty. The caller keeps
        separate series for each target, class, and horizon, and delays adding
        an observation until its forecast window has elapsed.
        """
