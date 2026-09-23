from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class TurnSpec(BaseModel):
    cmd: str
    prompt: str = "continue"
    max_tokens: int = 16


class JobSpec(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    name: str
    cls: Literal["interactive", "background"] = Field(alias="class", default="background")
    tenant: str = "t0"
    image: str | None = None
    repeat: int = 1
    setup: list[str | dict] = Field(default_factory=list)
    turns: list[TurnSpec]
    timeout_s: float = 1800.0
    deadline_s: float | None = None


class CollectionSpec(BaseModel):
    proxy_url: str | None = None
    model: str = "Qwen/Qwen3-0.6B"
    events_path: str = "runs/collect/events.jsonl"
    jobs: list[JobSpec]
    seed: int = 0
    concurrency: int = 1
