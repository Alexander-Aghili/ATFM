from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from atfm.board.state import SessionState
from atfm.schema.trace import TraceTable


class SessionPredictor(ABC):
    """Per-session predictor: session state in, samples out (D6 in the spec)."""

    name: str = "base"

    @abstractmethod
    def fit(self, train: TraceTable) -> "SessionPredictor": ...

    @abstractmethod
    def resumption(self, s: SessionState, now: float, n: int, rng: np.random.Generator) -> np.ndarray: ...

    @abstractmethod
    def next_call_isl(self, s: SessionState, n: int, rng: np.random.Generator) -> np.ndarray: ...

    @abstractmethod
    def spawn(self, s: SessionState, horizon: float, n: int, rng: np.random.Generator) -> np.ndarray: ...


class SeriesPredictor(ABC):
    """Aggregate time-series predictor (B0/B1): past demand values in, samples of the next out."""

    name: str = "series"

    @abstractmethod
    def predict(self, history: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray: ...
