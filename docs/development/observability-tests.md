# Tool-call observability for tests: Phoenix + OpenInference

Use Phoenix as an optional local trace and evaluation backend for ATFM tool
tests. The integration lives entirely in `atfm_experiments`; the runtime core
has no Phoenix, OpenInference, or OpenTelemetry imports or dependencies.
Saturation/planner changes are a separate workstream.

## Why this choice

Comparison made on 2026-09-27 from the primary documentation below, followed by
a hands-on Phoenix integration. This is a fit assessment, not a cross-vendor
performance benchmark. The immediate requirements are local repeatable tests,
no model/API credentials, tool-call correlation, progress events, error status,
queryable traces, and deterministic evaluation assertions.

| Candidate | Strength for our tests | Local setup / tradeoff | Decision |
| --- | --- | --- | --- |
| Phoenix + OpenInference | Standard trace instrumentation, tool spans, Python trace queries, evaluation annotations. | A local server with SQLite is sufficient; server Python dependencies still need isolation. | Selected for this test workflow. |
| Langfuse | Broad session/agent observability, OpenTelemetry support, self-hosted analytics. | Local deployment includes web/worker, PostgreSQL, ClickHouse, Redis/Valkey, and blob storage. | Strong alternative for broader analytics; more infrastructure than this fixture needs. |
| LangSmith | Framework-independent tracing plus evaluation and monitoring. | Its self-hosted offering is an Enterprise add-on with a license-key trial. | Reconsider when an existing organization deployment makes it the natural backend. |
| Braintrust | Tool/LLM/task traces connected to scoring and evaluation datasets. | Attractive evaluation workflow, but no additional requirement here justified adopting another platform over the tested local Phoenix path. | Not selected; no comparative ingestion or performance trial was run. |

Sources: [Phoenix architecture and SQLite](https://arize.com/docs/phoenix/self-hosting/deployment),
[Phoenix client query example](https://github.com/Arize-ai/phoenix/blob/main/packages/phoenix-client/examples/datasets/create_dataset_from_spans_example.py),
[Langfuse deployment architecture](https://langfuse.com/self-hosting),
[LangSmith self-hosting](https://docs.langchain.com/langsmith/self-hosted),
and [Braintrust tracing and scoring model](https://www.braintrust.dev/docs/instrument).

None of these choices removes the need for timely lifecycle signals in a live
controller. This integration exports **completed test events after execution**.
It neither delivers in-flight progress to ATFM nor establishes production
telemetry overhead. See the [broader integration assessment](../research/2026-09-27-agent-observability.md)
for that distinction.

## Package and dependency boundaries

The optional `atfm-experiments[observability]` extra contains the Phoenix client,
OpenInference semantic conventions, OpenTelemetry SDK, and OTLP/HTTP exporter.
It does not install the Phoenix server. Existing root-lock dependency versions
were preserved, including OpenAI and LiteLLM.

The server is pinned to Phoenix 20.16.0 in the separate uv project
`experiments/observability/server`, with its own lockfile and environment. The
server currently depends on an agent SDK that requires a newer OpenAI package
than the main workspace's existing stack. Isolating the process avoids changing
our inference/harness dependencies just to inspect test traces. The server is
not a member of the root workspace; its extra SDKs are not core dependencies.

## Run the test workflow

From the repository root, create an isolated client test environment and prepare
the separate server environment:

```bash
ATFM_OBS_ENV="$PWD/runs/observability-venv"
UV_PROJECT_ENVIRONMENT="$ATFM_OBS_ENV" uv sync \
  --all-packages --extra dev --extra observability --frozen
uv sync --project experiments/observability/server --frozen

ATFM_TEST_PHOENIX=1 "$ATFM_OBS_ENV/bin/pytest" -q \
  tests/experiments/test_tool_traces.py \
  tests/experiments/test_phoenix_integration.py \
  tests/test_package_boundaries.py

"$ATFM_OBS_ENV/bin/python" -m atfm_experiments.phoenix_smoke \
  --out runs/phoenix-smoke
```

The smoke command starts a disposable Phoenix/SQLite server, runs two real local
Python subprocesses through the existing sidecar, exports their events over
OTLP/HTTP, queries persisted spans, writes and reads evaluation annotations,
and stops the server. It requires `uv` and a POSIX host for process-group cleanup.
No GPU, LLM provider, cloud account, or credentials are required.

One subprocess exits successfully; the other intentionally exits with status 7.
Each prints synthetic pytest-style progress. Assertions check session/call
parentage, successful/error status, progress values and totals, original start
and end times within database timestamp precision, and persisted code-generated
evaluation scores. A passing score means the **fixture contract** was preserved;
it does not mean the intentionally failing tool succeeded.

Outputs under `--out`:

- `summary.json`: counts and verification summary.
- `spans.json`: traces read back from Phoenix, including progress events.
- `annotations.json`: evaluation results read back from Phoenix.
- `server/server.log` and SQLite data: debugging artifacts from the local run.

The URL in a disposable run's summary is historical: the server stops when the
command exits. To inspect a persistent development Phoenix instance, run the
same smoke command with `--endpoint http://127.0.0.1:6006`; it creates a unique
project without taking ownership of that server. Manage that instance with the
[Phoenix CLI](https://arize.com/docs/phoenix/self-hosting/deployment-options).

The automatic test server binds HTTP to loopback, disables its optional assistant,
MCP server, and web telemetry, and uses a separate data directory. Phoenix also
opens its gRPC listener on an ephemeral port with the upstream default binding;
this helper is a disposable test fixture, not a production deployment template.

## Reuse with other tool tests

`atfm_experiments.tool_traces.export_tool_events(events, tracer)` accepts the
existing typed ATFM event stream and an explicitly supplied tracer. Collect
with `InMemoryBus` or load events with `read_events`, then export after execution.
A caller-owned provider determines the exporter, batching, sampling, and SDK
span/event limits. Long progress histories must configure those limits rather
than assuming the SDK retains unlimited events.

The adapter creates one root per session covering the supplied tool calls and
one `TOOL` child per call. That root is not a complete agent lifetime or an
LLM-call graph. Calls are keyed by `(session_id, call_id)`. Failed calls mark
both the tool and session root as errors. Progress and data observations retain
their event timestamps; unknown totals remain absent. No commands, argument
payloads, prompts, or output text are captured.

It validates the complete stream before exporting: duplicate starts, missing
starts/ends, post-completion updates, nonfinite/negative timestamps, and reversed
per-call times fail explicitly. Cross-call interleaving is allowed. This is
intentionally stricter than live fail-open instrumentation: malformed test data
should fail the test rather than silently appear as a complete trace.

The adapter does not set the global tracer provider or monkeypatch any core
function. The smoke runner uses synchronous export outside measured execution;
do not wrap CPU benchmark timings around the export/query phase. Standard test
runs do not start a server, and tests needing optional packages skip when they
are absent. `ATFM_TEST_PHOENIX=1` explicitly opts into the server round trip.

## Verified evidence

The optional adapter/integration/boundary checks passed (9 tests), including a
real local OTLP ingestion and SQLite read-back. A clean default environment with
no observability packages passed 297 tests with 6 skips and the existing five
empty-metric warnings. This verifies both the opt-in path and the absence of a
new default dependency requirement.

The [saved smoke evidence](../research/results/phoenix-2026-09-27/summary.json)
records one session, two tool calls, one expected failure, six progress updates,
three persisted spans, and two evaluation annotations. The
[span data](../research/results/phoenix-2026-09-27/spans.json) and
[annotations](../research/results/phoenix-2026-09-27/annotations.json) are synthetic.
These checks validate trace semantics and local integration, not telemetry
throughput or forecast accuracy. Keep saturation experiments and subsequent
planner redesign separately measured.
