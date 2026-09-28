from __future__ import annotations

import json
import math
from itertools import groupby
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

Class = Literal["interactive", "background"]


def is_missing_scalar(value) -> bool:
    """Recognize the None/float-NaN sentinels used by canonical trace records."""
    return value is None or (isinstance(value, float) and math.isnan(value))


class ProgressEvent(BaseModel):
    t: float
    completed: float
    total: float | None = None
    phase: str | None = None


class DataEvent(BaseModel):
    t: float
    metric: str
    value: float


class TraceRow(BaseModel):
    """One LLM call plus the tool phase this call launched (if any)."""

    model_config = ConfigDict(populate_by_name=True)
    session_id: str
    parent_session_id: str | None = None
    cls: Class = Field(alias="class")
    tenant: str
    turn_index: int
    t_request: float
    t_first_token: float | None = None
    t_last_token: float | None = None
    isl: int
    osl: int
    prefix_hit_tokens: int = 0
    kv_blocks_by_tier: dict[str, int] = Field(default_factory=dict)
    worker_id: str | None = None
    tool_name: str | None = None
    tool_args_hash: str | None = None
    t_tool_start: float | None = None
    t_tool_end: float | None = None
    tool_exit_status: int | None = None
    progress_events: list[ProgressEvent] = Field(default_factory=list)
    data_events: list[DataEvent] = Field(default_factory=list)
    backend_id: str | None = None
    spawned_children: int = 0
    perturbation_flag: bool = False
    source: str


TRACE_COLUMNS: list[str] = [
    "session_id", "parent_session_id", "class", "tenant", "turn_index",
    "t_request", "t_first_token", "t_last_token", "isl", "osl", "prefix_hit_tokens",
    "kv_blocks_by_tier", "worker_id", "tool_name", "tool_args_hash",
    "t_tool_start", "t_tool_end", "tool_exit_status", "progress_events", "data_events",
    "backend_id", "spawned_children", "perturbation_flag", "source",
]

NESTED_COLUMNS = ("kv_blocks_by_tier", "progress_events", "data_events")


class TraceTable:
    def __init__(self, df: pd.DataFrame):
        missing = [c for c in TRACE_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"trace table missing columns: {missing}")
        self.df = (
            df[TRACE_COLUMNS]
            .sort_values(["session_id", "t_request"], kind="stable")
            .reset_index(drop=True)
        )

    @classmethod
    def from_rows(cls, rows: list[TraceRow]) -> "TraceTable":
        recs = [r.model_dump(by_alias=True) for r in rows]
        df = pd.DataFrame.from_records(recs, columns=TRACE_COLUMNS)
        return cls(df)

    def to_parquet(self, path: str | Path) -> None:
        # Nested columns are stored as JSON strings: parquet cannot hold empty structs.
        df = self.df.copy()
        for col in NESTED_COLUMNS:
            df[col] = df[col].apply(json.dumps)
        df.to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: str | Path) -> "TraceTable":
        df = pd.read_parquet(path)
        for col in NESTED_COLUMNS:
            df[col] = df[col].apply(lambda v: json.loads(v) if isinstance(v, str) else v)
        return cls(df)

    def sessions(self) -> Iterator[tuple[str, pd.DataFrame]]:
        for sid, g in self.df.groupby("session_id", sort=False):
            yield sid, g

    def session_records(self, batch_size: int = 4096) -> Iterator[tuple[str, list[dict]]]:
        """Yield session records in canonical stable order without per-session DataFrames.

        Conversion is batched to bound temporary memory by a batch plus the
        largest session. Nested event containers are shared with the table;
        consumers must not mutate them. The table must retain its canonical
        session/time ordering, established by construction.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        records = (
            row
            for start in range(0, len(self.df), batch_size)
            for row in self.df.iloc[start:start + batch_size].to_dict("records")
        )
        for sid, rows in groupby(records, key=lambda row: row["session_id"]):
            yield sid, list(rows)

    def time_range(self) -> tuple[float, float]:
        return float(self.df["t_request"].min()), float(self.df["t_request"].max())

    def kv_blocks(self, block_size: int = 16) -> np.ndarray:
        return np.ceil(self.df["isl"].to_numpy() / block_size).astype(int)

    def __len__(self) -> int:
        return len(self.df)
