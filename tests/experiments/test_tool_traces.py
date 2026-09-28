import pytest

pytest.importorskip("openinference.semconv")
pytest.importorskip("opentelemetry.sdk")

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode

from atfm.schema.events import ToolStart, ToolEnd, ToolProgress, ToolData
from atfm_experiments.tool_traces import export_tool_events


@pytest.fixture
def tracing():
    exporter = InMemorySpanExporter()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider.get_tracer("test"), exporter
    provider.shutdown()


def events(sid="s", code=0):
    return [ToolStart(t=1, session_id=sid, call_id="same", turn_index=0, tool_name="test"),
            ToolProgress(t=2, session_id=sid, call_id="same", completed=1),
            ToolData(t=2.5, session_id=sid, call_id="same", metric="rows", value=4),
            ToolEnd(t=3, session_id=sid, call_id="same", exit_status=code)]


def test_timestamps_progress_errors_and_session_scoping(tracing):
    tracer, exporter = tracing
    global_provider = trace.get_tracer_provider()
    counts = export_tool_events(events() + events("other", 1), tracer)
    spans = exporter.get_finished_spans()
    assert counts == dict(sessions=2, tools=2, failures=1, progress_events=2, data_events=2)
    tools = [s for s in spans if s.name == "test"]
    roots = {s.context.span_id: s for s in spans if s.parent is None}
    assert len(roots) == 2
    for tool in tools:
        assert (tool.start_time, tool.end_time) == (10**9, 3 * 10**9)
        assert tool.context.trace_id == roots[tool.parent.span_id].context.trace_id
        assert tool.events[0].timestamp == 2 * 10**9
        assert "total" not in tool.events[0].attributes
        assert tool.events[1].attributes["metric"] == "rows"
        assert not any(k in tool.attributes for k in ("input.value", "output.value", "tool.parameters"))
    assert {s.status.status_code for s in tools} == {StatusCode.OK, StatusCode.ERROR}
    assert trace.get_tracer_provider() is global_provider


@pytest.mark.parametrize("case", ["missing_end", "missing_start", "duplicate", "reversed", "late_progress", "nan"])
def test_bad_stream_fails_before_export(tracing, case):
    tracer, exporter = tracing
    stream = events()
    if case == "missing_end":
        stream.pop()
    elif case == "missing_start":
        stream.pop(0)
    elif case == "duplicate":
        stream.insert(1, stream[0])
    elif case == "reversed":
        stream[-1] = stream[-1].model_copy(update={"t": 1.5})
    elif case == "late_progress":
        stream.append(stream[1])
    else:
        stream[0] = stream[0].model_copy(update={"t": float("nan")})
    with pytest.raises(ValueError):
        export_tool_events(stream, tracer)
    assert exporter.get_finished_spans() == ()
