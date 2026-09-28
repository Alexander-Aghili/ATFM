# Existing tool-call and agent analytics infrastructure

Reviewed 2026-09-27 against the primary documentation linked below. This is an
integration assessment, not a measured product comparison or a claim of feature
parity. No vendor SDK, hosted service, or telemetry export was enabled by this
review.

## Existing layers

| Layer | Examples and documented capabilities | Potential role for ATFM |
| --- | --- | --- |
| Trace instrumentation and transport | [OpenTelemetry GenAI agent/tool conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/main/docs/gen-ai/gen-ai-agent-spans.md); [OpenInference](https://arize.com/docs/phoenix/resources/python-api) provides agent/tool decorators and framework instrumentors. | Reuse correlation and instrumentation conventions rather than creating another incompatible tracing format. |
| Observability and analytics | [Langfuse](https://langfuse.com/docs) captures LLM and non-LLM operations, sessions, and agent graphs; supports OpenTelemetry and self-hosting. | Inspect tool durations, session histories, costs, errors, and forecast outcomes. |
| Observability plus evaluation | [Phoenix](https://arize.com/docs/phoenix/) combines tracing, evaluation, datasets, and experiments; accepts instrumented application traces. | A candidate backend for open-source trace analysis and offline evaluation. |
| Agent tracing and monitoring | [LangSmith](https://www.langchain.com/langsmith/observability) supports framework-independent instrumentation, tool/agent trajectories, and OpenTelemetry integration. | Reuse an organization's existing traces and workflow debugging instead of requiring a new analytics UI. |
| Tracing connected to evaluations | [Braintrust](https://www.braintrust.dev/docs/instrument) represents tool, LLM, task, and scoring spans with timing, metadata, usage, and errors. | Analyze failed tool trajectories and convert observed behavior into evaluation data. |
| Agent orchestration | [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview) is a runtime for long-running stateful agents, with persistence, streaming, and human intervention. | A source of lifecycle and spawn events; it is a different layer from ATFM's shared serving-resource control. |
| Tool interoperability and progress | [MCP progress notifications](https://modelcontextprotocol.io/specification/2025-11-25/basic/utilities/progress) carry a correlated progress value and optional total/message. | A candidate input for progress-conditioned remaining-time forecasts. Progress is optional and must not be assumed available for every tool. |

Langfuse also documents [agent workflow graphs](https://langfuse.com/docs/observability/features/agent-graphs)
and a [self-hosted ingestion architecture](https://langfuse.com/self-hosting).
Those are useful analytics capabilities, but do not themselves establish a
latency bound suitable for making ATFM admission decisions.

## The difference between a trace and a live control signal

A completed tool span can supply historical duration, identity, errors, and
parent-child relationships. It cannot by itself tell a controller, during the
operation, that the tool has just started or reached a particular milestone.
The standard OpenTelemetry simple and batch processors export **finished**
spans; a custom processor can receive `OnStart`, but that synchronous callback
must not block. These distinctions follow from the
[OpenTelemetry tracing SDK](https://opentelemetry.io/docs/specs/otel/trace/sdk/).

Our integration recommendation is therefore to emit small lifecycle/progress
events promptly to ATFM and export correlated full traces asynchronously to an
existing analytics backend. This is an architectural recommendation, not a
claim that every vendor lacks in-flight trace features. Measure delivery delay,
dropped events, ordering, retries, and sampling behavior for the chosen backend.
A completed span's embedded progress events do not become live notifications
merely because they contain earlier timestamps.

MCP provides a useful transport for progress, but a server may decline to send
updates or omit totals. A progress counter also need not correspond linearly to
remaining time. ATFM still needs duration fallbacks and tool-specific progress
calibration. The linked MCP page documents revision 2025-11-25; negotiate the
actual deployment's protocol version rather than assuming all implementations
support the same features.

## Suggested ATFM boundary

Keep the existing typed event API as the controller's internal contract and
add a small versioned adapter at the instrumentation boundary. GenAI semantic
conventions have moved into their own repository; pin mappings to a known
version and test them instead of treating attribute names as permanent.
The [OpenTelemetry registry](https://opentelemetry.io/docs/specs/semconv/registry/attributes/gen-ai/)
links the current convention location.

| Existing ATFM event / field | Candidate source | Additional work needed |
| --- | --- | --- |
| `session.start`, session/parent IDs, tenant, traffic class | Agent runtime start hooks and explicit application metadata | Define session lifetime and cross-process propagation; trace ID is not automatically a long-lived session ID. |
| `llm.request`, `llm.done`, request ID and token counts | Model instrumentation hooks / completed LLM spans | Emit request start promptly; normalize provider usage and preserve request correlation. |
| `llm.first_token` | Streaming client hook | Ordinary completed-call duration is insufficient for TTFT. |
| `tool.start`, `tool.end`, call ID, tool/backend identity | Tool wrapper or framework hooks | Correlate attempts and distinguish retry, cancellation, and failure. |
| `tool.progress`, `tool.data` | MCP progress or existing sidecar parsers | Preserve units, unknown totals, timestamps, and call identity; calibrate their meaning. |
| `spawn.request` | Runtime subagent/task creation hook | Preserve parent and child session identity across workers. |
| `worker.metrics` and KV residency | Serving-engine telemetry | Application traces alone do not establish worker capacity or actual cache residency. |

The repository already has sidecar lifecycle/progress events and several agent
adapters. This review did not find an existing OpenTelemetry/Langfuse/Phoenix/
LangSmith importer in the core. The mapping above is proposed work, not an
implemented integration.

For a first integration, shortlist **Phoenix/OpenInference or Langfuse** for
trace inspection and evaluation, and use whichever agent runtime the workload
already has. Compare them on actual framework coverage, export/query access,
self-hosting needs, and telemetry overhead. Do not duplicate their dashboards
as the first ATFM task. ATFM's distinct question remains whether timely tool
signals improve future-demand predictions and shared-worker admission/cache
decisions; tracing quality alone does not demonstrate that benefit.
