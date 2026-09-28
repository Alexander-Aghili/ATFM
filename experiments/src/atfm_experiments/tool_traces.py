"""Optional, offline OpenInference export of completed ATFM tool-test events.

This adapter does not instrument the core, set a global tracer, or transport
live control signals. Export after timing measurements to keep them meaningful.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math
from typing import Iterable

from openinference.semconv.trace import SpanAttributes as Attr
from opentelemetry.context import Context
from opentelemetry.trace import StatusCode, Tracer, set_span_in_context

from atfm.schema.events import Event, ToolData, ToolEnd, ToolProgress, ToolStart


@dataclass
class _Call:
    start: ToolStart
    end: ToolEnd | None = None
    updates: list[ToolProgress | ToolData] = field(default_factory=list)


def _calls(events: Iterable[Event]) -> list[_Call]:
    calls: dict[tuple[str, str], _Call] = {}
    for event in events:
        if not isinstance(event, (ToolStart, ToolEnd, ToolProgress, ToolData)):
            continue
        if not math.isfinite(event.t) or event.t < 0:
            raise ValueError("tool event timestamps must be finite and nonnegative")
        key = event.session_id, event.call_id
        if isinstance(event, ToolStart):
            if key in calls:
                raise ValueError(f"duplicate tool start: {key}")
            calls[key] = _Call(event)
            continue
        call = calls.get(key)
        if call is None or call.end is not None:
            raise ValueError(f"unmatched or already completed tool call: {key}")
        last_time = call.updates[-1].t if call.updates else call.start.t
        if event.t < last_time:
            raise ValueError(f"nonmonotonic tool timestamps: {key}")
        if isinstance(event, ToolEnd):
            call.end = event
        else:
            call.updates.append(event)
    if any(call.end is None for call in calls.values()):
        raise ValueError("offline export requires completed tool calls")
    return list(calls.values())


def export_tool_events(events: Iterable[Event], tracer: Tracer) -> dict[str, int]:
    """Validate then export session roots and TOOL children with original timestamps.

    Session roots cover only the supplied tool events, not the agent's complete
    lifetime. Progress is preserved as span events with optional totals omitted
    when unknown. Prompts, commands, arguments, and output text are not captured.
    Calls are correlated by both session ID and call ID. Malformed/incomplete
    streams fail before emitting any spans rather than inventing completion.
    """
    sessions: dict[str, list[_Call]] = defaultdict(list)
    for call in _calls(events):
        sessions[call.start.session_id].append(call)
    counts = dict(sessions=len(sessions), tools=0, failures=0, progress_events=0, data_events=0)
    for sid, calls in sessions.items():
        root = tracer.start_span("atfm.tool_test", context=Context(),
                                 start_time=int(min(c.start.t for c in calls) * 1e9),
                                 attributes={Attr.OPENINFERENCE_SPAN_KIND: "CHAIN", Attr.SESSION_ID: sid,
                                             "atfm.trace_scope": "completed_tool_events"})
        context = set_span_in_context(root)
        try:
            for call in calls:
                start, end = call.start, call.end
                assert end is not None
                span = tracer.start_span(start.tool_name, context=context, start_time=int(start.t * 1e9),
                                         attributes={Attr.OPENINFERENCE_SPAN_KIND: "TOOL", Attr.SESSION_ID: sid,
                                                     Attr.TOOL_NAME: start.tool_name, Attr.TOOL_ID: start.call_id,
                                                     "atfm.backend_id": start.backend_id,
                                                     "atfm.turn_index": start.turn_index,
                                                     "atfm.exit_status": end.exit_status,
                                                     "atfm.output_chars": end.output_chars})
                try:
                    for update in call.updates:
                        attributes = update.model_dump(exclude={"kind", "t", "session_id", "call_id"}, exclude_none=True)
                        span.add_event(update.kind, attributes=attributes, timestamp=int(update.t * 1e9))
                        counts["progress_events" if isinstance(update, ToolProgress) else "data_events"] += 1
                    failed = end.exit_status != 0
                    span.set_status(StatusCode.ERROR if failed else StatusCode.OK)
                    if failed:
                        root.set_status(StatusCode.ERROR)
                    counts["tools"] += 1
                    counts["failures"] += int(failed)
                finally:
                    span.end(end_time=int(end.t * 1e9))
        finally:
            root.end(end_time=int(max(c.end.t for c in calls if c.end is not None) * 1e9))
    return counts
