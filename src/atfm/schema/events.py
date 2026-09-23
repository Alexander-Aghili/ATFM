from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter


class _E(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    t: float


class SessionStart(_E):
    kind: Literal["session.start"] = "session.start"
    session_id: str
    tenant: str
    cls: Literal["interactive", "background"] = Field(alias="class")
    parent_session_id: str | None = None
    deadline: float | None = None


class LlmRequest(_E):
    kind: Literal["llm.request"] = "llm.request"
    session_id: str
    turn_index: int
    request_id: str
    isl: int
    predicted_osl: int | None = None
    hints: dict = Field(default_factory=dict)
    held_s: float = 0.0


class LlmFirstToken(_E):
    kind: Literal["llm.first_token"] = "llm.first_token"
    session_id: str
    request_id: str


class LlmDone(_E):
    kind: Literal["llm.done"] = "llm.done"
    session_id: str
    request_id: str
    osl: int = 0
    worker_id: str | None = None
    prefix_hit_tokens: int = 0
    status: int = 200


class ToolStart(_E):
    kind: Literal["tool.start"] = "tool.start"
    session_id: str
    turn_index: int
    call_id: str
    tool_name: str
    backend_id: str = "local"
    args_hash: str | None = None


class ToolProgress(_E):
    kind: Literal["tool.progress"] = "tool.progress"
    session_id: str
    call_id: str
    completed: float
    total: float | None = None
    phase: str | None = None


class ToolData(_E):
    kind: Literal["tool.data"] = "tool.data"
    session_id: str
    call_id: str
    metric: str
    value: float


class ToolEnd(_E):
    kind: Literal["tool.end"] = "tool.end"
    session_id: str
    call_id: str
    exit_status: int
    output_chars: int = 0


class SpawnRequest(_E):
    kind: Literal["spawn.request"] = "spawn.request"
    parent_session_id: str
    child_session_id: str


class WorkerMetrics(_E):
    kind: Literal["worker.metrics"] = "worker.metrics"
    worker_id: str
    kv_blocks_used: int
    kv_blocks_total: int
    queue_depth: int = 0
    tier_blocks: dict[str, int] = Field(default_factory=dict)


Event = Annotated[
    Union[SessionStart, LlmRequest, LlmFirstToken, LlmDone, ToolStart, ToolProgress, ToolData, ToolEnd,
          SpawnRequest, WorkerMetrics],
    Field(discriminator="kind"),
]
_adapter = TypeAdapter(Event)


def parse_event(d: dict) -> Event:
    return _adapter.validate_python(d)


def event_to_dict(e: Event) -> dict:
    return e.model_dump(by_alias=True)
