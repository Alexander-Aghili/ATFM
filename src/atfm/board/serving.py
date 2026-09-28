"""Constant-time prediction reads with explicit freshness and legacy fallback."""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

from fastapi.responses import JSONResponse, Response

from atfm.board.execution import StageTimings
from atfm.board.live import LiveBoard
from atfm.board.publication import PredictionView, Publication, PublishedBoard, snapshot_json
from atfm.proxy.prediction import PredictionRunner


class BoardReader:
    def __init__(self, board, now, budget_s, max_age_s, monotonic=time.monotonic):
        if not math.isfinite(max_age_s) or max_age_s <= 0:
            raise ValueError('prediction_max_age_s must be finite and positive')
        self.budget_s, self.max_age_s, self.monotonic = budget_s, max_age_s, monotonic
        self.timings = StageTimings()
        self.standard = (type(board).expected_service is LiveBoard.expected_service
                         and type(board).expected_tool_next is LiveBoard.expected_tool_next)
        self.legacy = None if self.standard else PredictionRunner(board, 4, budget_s)
        captured_at = monotonic()
        initial = self.capture(board, now)
        self.published = PublishedBoard(Publication(0, captured_at, None, initial, snapshot_json(None)))
        self.counts = dict(served=0, stale=0, unavailable=0)

    def capture(self, board, now):
        return PredictionView.capture(board, now) if self.standard else None

    async def predict(self, data):
        start = self.monotonic()
        with self.timings.measure('prediction_read'):
            publication = self.published.read()
        age = max(0., start - publication.captured_at)
        stale = age > self.max_age_s
        result = None if stale else await self._values(publication, data)
        elapsed = self.monotonic() - start
        unavailable = result is None or elapsed > self.budget_s
        self.counts['stale' if stale else 'unavailable' if unavailable else 'served'] += 1
        with self.timings.measure('prediction_serialize'):
            return JSONResponse(dict(e_service_s=0. if unavailable else result[0],
                                     e_tool_next_s=0. if unavailable else result[1], elapsed_ms=elapsed * 1000.,
                                     over_budget=unavailable, stale=stale, prediction_age_s=age,
                                     prediction_version=publication.version))

    async def _values(self, publication, data):
        with self.timings.measure('prediction_compute'):
            sid, isl, osl = data.get('session_id', ''), int(data.get('isl', 0)), int(data.get('osl', 0))
            if self.standard:
                return publication.predictions.predict(sid, isl, osl)
            meta = SimpleNamespace(session_id=sid, isl=isl, predicted_osl=osl)
            return await self.legacy.predict(meta, None)

    def snapshot(self):
        return Response(self.published.read().snapshot_json, media_type='application/json')

    def status(self):
        publication = self.published.read()
        return dict(version=publication.version, age_s=max(0., self.monotonic() - publication.captured_at),
                    max_age_s=self.max_age_s, sessions=len(publication.predictions.tools) if self.standard else None,
                    snapshot_t=None if publication.snapshot is None else publication.snapshot.t,
                    predictions=dict(self.counts), stages=self.timings.snapshot())

    def close(self):
        if self.legacy is not None:
            self.legacy.close()
