from __future__ import annotations

import re
from typing import Protocol


class Parser(Protocol):
    def feed(self, line: str, t: float) -> dict | None: ...


_COLLECTED = re.compile(r"collected (\d+) items?")
_RESULT = re.compile(r"::\S.*\b(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")
_SUMMARY = re.compile(r"^=+ .*(passed|failed|error|skipped|no tests ran).* in [\d.]+s")
_PERCENT = re.compile(r"\[\s*(\d{1,3})%\]")
_COUNTER = re.compile(r"(?<![\d.])(\d+)\s*/\s*(\d+)(?![\d.])")


class PytestParser:
    """Strong signal: collected N items, then one line per test result, then the summary line."""

    def __init__(self):
        self.total: int | None = None
        self.done = 0

    def feed(self, line: str, t: float) -> dict | None:
        m = _COLLECTED.search(line)
        if m:
            self.total = int(m.group(1))
            self.done = 0
            return {"completed": 0, "total": self.total, "phase": "collect"}
        if _RESULT.search(line):
            self.done += 1
            return {"completed": self.done, "total": self.total, "phase": "run"}
        if _SUMMARY.search(line):
            total = self.total if self.total is not None else self.done
            return {"completed": total, "total": total, "phase": "done"}
        return None


class PercentParser:
    """cmake/make style `[ 45%]` markers (also pytest -v)."""

    def feed(self, line: str, t: float) -> dict | None:
        m = _PERCENT.search(line)
        if not m:
            return None
        return {"completed": float(m.group(1)), "total": 100.0, "phase": "run"}


class CounterParser:
    """Generic k/N counters (ninja `[12/345]`, `processed 10/100 rows`)."""

    def feed(self, line: str, t: float) -> dict | None:
        m = _COUNTER.search(line)
        if not m:
            return None
        k, n = float(m.group(1)), float(m.group(2))
        if n < 3 or k > n:
            return None
        return {"completed": k, "total": n, "phase": "run"}


_ROWS_A = re.compile(r"\b(\d+)\s*/\s*(\d+)\s+rows?\b|\brows?\s+processed\s+(\d+)\s*/\s*(\d+)", re.I)
_ROWS_B = re.compile(r"\bprocessed\s+(\d+)\s+of\s+(\d+)\s+rows?\b", re.I)
_STAGE = re.compile(r"\bstage\s+(\d+)\s*(?:/|of)\s*(\d+)\b", re.I)
_STEP = re.compile(r"\b(step|epoch|iter(?:ation)?)\s+(\d+)\s*(?:/|of)\s*(\d+)\b", re.I)
_LOSS = re.compile(r"\bloss\s*[=:]\s*([-+0-9.eE]+)", re.I)


def _kn(k: float, n: float, phase: str) -> dict | None:
    if n <= 0 or k > n:
        return None
    return {"completed": float(k), "total": float(n), "phase": phase}


class RowsParser:
    """dbt / Spark / ETL style row counters: `rows processed k/N`, `processed k of N rows`, `k/N rows`."""

    def feed(self, line: str, t: float) -> dict | None:
        m = _ROWS_A.search(line)
        if m:
            k, n = (m.group(1), m.group(2)) if m.group(1) else (m.group(3), m.group(4))
            return _kn(float(k), float(n), "rows")
        m = _ROWS_B.search(line)
        return _kn(float(m.group(1)), float(m.group(2)), "rows") if m else None


class StageParser:
    """Pipeline stage markers: `stage k/N`, `Stage k of N`."""

    def feed(self, line: str, t: float) -> dict | None:
        m = _STAGE.search(line)
        return _kn(float(m.group(1)), float(m.group(2)), "stage") if m else None


class TrainingParser:
    """Training monitors: `step k/N` or `epoch k/N` as progress, `loss=x` or `loss: x` as a data event."""

    def feed(self, line: str, t: float) -> dict | None:
        m = _STEP.search(line)
        return _kn(float(m.group(2)), float(m.group(3)), m.group(1).lower()[:5]) if m else None

    def feed_data(self, line: str, t: float) -> dict | None:
        m = _LOSS.search(line)
        if not m:
            return None
        try:
            return {"metric": "loss", "value": float(m.group(1))}
        except ValueError:
            return None


class RateParser:
    """Weak signal: output line rate over a window, emitted as a data event."""

    def __init__(self, window_s: float = 5.0):
        self.window_s = window_s
        self.t0: float | None = None
        self.n = 0

    def feed(self, line: str, t: float) -> dict | None:
        return None

    def feed_data(self, line: str, t: float) -> dict | None:
        if self.t0 is None:
            self.t0 = t
        self.n += 1
        if t - self.t0 > self.window_s:
            v = self.n / (t - self.t0)
            self.t0, self.n = t, 0
            return {"metric": "lines_per_s", "value": v}
        return None


def default_parsers() -> list:
    return [PytestParser(), PercentParser(), RowsParser(), StageParser(), TrainingParser(), CounterParser(), RateParser()]
