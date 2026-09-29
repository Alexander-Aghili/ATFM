"""Predeclared search space, workload contexts and independent validation budget."""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

from .config import LoadConfig


class Settings(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    proxy_window: int = Field(ge=1, strict=True)
    upstream_pool_shards: int | None = Field(default=None, ge=1, le=100, strict=True)
    prediction_limit: int = Field(default=4, ge=1, strict=True)
    prediction_budget_s: float = Field(default=.05, gt=0)


class Constraints(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    max_p95_s: float | None = Field(default=None, gt=0)
    max_cpu_s_per_request: float | None = Field(default=None, gt=0)
    min_prediction_use: float | None = Field(default=None, ge=0, le=1)


class TuningPlan(BaseModel):
    model_config = ConfigDict(extra='forbid', allow_inf_nan=False)
    workloads: dict[str, LoadConfig] = Field(min_length=1, max_length=16)
    candidates: dict[str, Settings] = Field(min_length=2, max_length=32)
    baseline: str
    search_seeds: list[StrictInt] = Field(default_factory=lambda: [7, 8, 9], min_length=3, max_length=30)
    validation_seeds: list[StrictInt] = Field(default_factory=lambda: [101, 102, 103, 104, 105], min_length=3, max_length=30)
    objective: Literal['latency', 'cpu'] = 'latency'
    constraints: Constraints = Field(default_factory=Constraints)
    min_improvement: float = Field(default=.02, ge=0, lt=1)
    max_context_regression: float = Field(default=0., ge=0, lt=1)
    confidence: float = Field(default=.95, gt=.5, lt=1)
    bootstrap_samples: int = Field(default=2000, ge=100, le=10000)
    order_seed: int = Field(default=42, ge=0, strict=True)
    max_trials: int = Field(default=256, ge=1, le=10000)
    max_requests: int = Field(default=1_000_000, ge=1, le=10_000_000)

    @model_validator(mode='after')
    def validate_design(self):
        _validate_names(self)
        _validate_seeds(self.search_seeds, self.validation_seeds)
        copies = len(self.candidates) * len(self.search_seeds) + 2 * len(self.validation_seeds)
        if copies * len(self.workloads) > self.max_trials:
            raise ValueError('planned search and validation exceed max_trials')
        if copies * sum(w.sessions * w.turns for w in self.workloads.values()) > self.max_requests:
            raise ValueError('planned search and validation exceed max_requests')
        if any(w.profile_proxy for w in self.workloads.values()):
            raise ValueError('instrumented profiling is not valid tuning evidence')
        for workload in self.workloads.values():
            for candidate in self.candidates.values():
                LoadConfig.model_validate(workload.model_dump() | candidate.model_dump())
        return self


def _validate_names(plan):
    if plan.baseline not in plan.candidates:
        raise ValueError('baseline must name a candidate')
    if any(not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}', name) for name in [*plan.workloads, *plan.candidates]):
        raise ValueError('context and candidate names must be safe directory identifiers')
    values = [tuple(candidate.model_dump().values()) for candidate in plan.candidates.values()]
    if len(set(values)) != len(values):
        raise ValueError('candidate settings must be distinct')


def _validate_seeds(search, validation):
    if any(type(seed) is not int or seed < 0 for seed in search + validation):
        raise ValueError('seeds must be nonnegative integers')
    if len(set(search)) != len(search) or len(set(validation)) != len(validation):
        raise ValueError('seeds must be unique within each phase')
    if set(search) & set(validation):
        raise ValueError('search and validation seeds must be disjoint')
