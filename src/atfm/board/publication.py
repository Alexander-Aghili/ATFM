"""Immutable request-facing values, replaced only after a complete board tick."""
from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from types import MappingProxyType
from typing import Mapping

from fastapi.responses import JSONResponse

from atfm.board.forecaster import CLASSES, TARGETS
from atfm.board.live import LiveBoard


@dataclass(frozen=True)
class PredictionView:
    tools: Mapping[str, float | None]
    pooled: float | None
    prefill_tps: float
    decode_tps: float

    @classmethod
    def capture(cls, board, now):
        states = board.registry.states(now)
        dm = getattr(board.forecaster.predictor, 'dm', None)
        mean = lambda tool: tool_mean(dm, tool)
        means = {None: mean(None)}
        tools = {}
        for state in states:
            tool = LiveBoard.next_tool(state)
            if tool not in means:
                means[tool] = mean(tool)
            tools[state.session_id] = means[tool]
        return cls(MappingProxyType(tools), means[None], board.prefill_tps, board.decode_tps)

    def predict(self, sid, isl, osl):
        tool = self.tools.get(sid, self.pooled)
        return None if tool is None else (isl / self.prefill_tps + osl / self.decode_tps, tool)


@dataclass(frozen=True)
class Publication:
    version: int
    captured_at: float
    snapshot: object
    predictions: PredictionView | None
    snapshot_json: bytes


class PublishedBoard:
    def __init__(self, initial):
        self.lock, self.current = Lock(), initial

    def read(self):
        with self.lock:
            return self.current

    def publish(self, complete):
        with self.lock:
            self.current = complete


def snapshot_json(snapshot):
    if snapshot is None:
        return JSONResponse({'t': None, 'horizons': [], 'q50': {}, 'q90': {}}).body
    result = {'t': snapshot.t, 'model_id': snapshot.model_id, 'horizons': list(snapshot.horizons)}
    for label, q in (('q50', .5), ('q90', .9)):
        result[label] = {t: {c: snapshot.quantiles(t, c, q).tolist() for c in CLASSES} for t in TARGETS}
    result['endogenous_fraction'] = {c: snapshot.endogenous_fraction[c].tolist()
                                     if c in snapshot.endogenous_fraction else [] for c in CLASSES}
    return JSONResponse(result).body


def tool_mean(model, tool):
    try:
        return model.mean(tool) if model is not None else 0.
    except Exception:
        return None
