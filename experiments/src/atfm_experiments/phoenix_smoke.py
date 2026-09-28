"""Run sidecar tool fixtures, export to local Phoenix, and verify stored traces."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time
from uuid import uuid4

import httpx
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from phoenix.client import Client

from atfm.bus import InMemoryBus
from atfm.sidecar.core import ToolContext, run_tool
from atfm_experiments.tool_traces import export_tool_events


@contextmanager
def local_phoenix(directory: Path):
    """Own a disposable SQLite server; stop it even when an assertion fails."""
    directory.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    endpoint = f"http://127.0.0.1:{port}"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PHOENIX_", "OTEL_"))}
    env.pop("UV_PROJECT_ENVIRONMENT", None)
    env.update(PHOENIX_HOST="127.0.0.1", PHOENIX_PORT=str(port), PHOENIX_GRPC_PORT="0",
               PHOENIX_WORKING_DIR=str(directory.resolve()), PHOENIX_TELEMETRY_ENABLED="false",
               PHOENIX_ENABLE_MCP_SERVER="false", PHOENIX_DISABLE_AGENT_ASSISTANT="true")
    with (directory / "server.log").open("w") as log:
        server_project = Path(__file__).resolve().parents[2] / "observability" / "server"
        proc = subprocess.Popen(["uv", "run", "--project", str(server_project), "--frozen", "phoenix", "serve"], env=env,
                                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            deadline = time.monotonic() + 60
            with httpx.Client(base_url=endpoint, trust_env=False, timeout=1) as client:
                while time.monotonic() < deadline:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Phoenix exited; see {directory / 'server.log'}")
                    try:
                        if client.get("/healthz").is_success:
                            break
                    except httpx.TransportError:
                        pass
                    time.sleep(0.2)
                else:
                    raise TimeoutError(f"Phoenix startup timed out; see {directory / 'server.log'}")
            yield endpoint
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()


def smoke(endpoint: str, out: Path) -> dict:
    """Synthetic local calls only; no model provider or production trace required."""
    out.mkdir(parents=True, exist_ok=True)
    project = f"atfm-tools-{uuid4().hex}"
    bus = InMemoryBus()
    for index, code in enumerate((0, 7)):
        script = "print('collected 2 items'); print('a::test PASSED'); print('b::test PASSED'); " + f"raise SystemExit({code})"
        result = run_tool([sys.executable, "-u", "-c", script],
                          ToolContext("smoke-session", index, f"fixture-{index}"), bus, shell=False)
        assert result.returncode == code
    events = bus.drain()
    provider = TracerProvider(resource=Resource.create({"service.name": "atfm-tests", "openinference.project.name": project}),
                              sampler=ALWAYS_ON, shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint=endpoint.rstrip("/") + "/v1/traces",
                                                                    headers={}, timeout=5)))
    try:
        counts = export_tool_events(events, provider.get_tracer("atfm.experiments.tools"))
        assert provider.force_flush(timeout_millis=5000)
    finally:
        provider.shutdown()
    with httpx.Client(base_url=endpoint, trust_env=False, timeout=5) as http:
        client = Client(http_client=http)
        deadline = time.monotonic() + 20
        spans = []
        while time.monotonic() < deadline:
            try:
                spans = client.spans.get_spans(project_identifier=project, limit=10)
                if len(spans) == 3:
                    break
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 404:
                    raise
            time.sleep(0.2)
        assert len(spans) == 3, f"expected three persisted spans, got {len(spans)}"
        tools = [s for s in spans if s["span_kind"] == "TOOL"]
        root = next(s for s in spans if s["span_kind"] == "CHAIN")
        assert len(tools) == 2
        assert {s["status_code"] for s in tools} == {"OK", "ERROR"}
        for tool in tools:
            assert tool["parent_id"] == root["context"]["span_id"]
            assert tool["context"]["trace_id"] == root["context"]["trace_id"]
            assert len(tool.get("events", [])) == 3
            assert datetime.fromisoformat(tool["end_time"]) >= datetime.fromisoformat(tool["start_time"])
            client.spans.add_span_annotation(span_id=tool["context"]["span_id"], annotation_name="fixture_contract",
                                             annotator_kind="CODE", label="pass", score=1.0, sync=True)
        annotations = client.spans.get_span_annotations(spans=tools, project_identifier=project)
        assert len(annotations) == 2
        summary = dict(project=project, **counts, persisted_spans=len(spans), verified_annotations=len(annotations),
                       ui_url=endpoint, checks="parentage, status, progress count, timestamps, annotation persistence")
        (out / "spans.json").write_text(json.dumps(spans, indent=2))
        (out / "summary.json").write_text(json.dumps(summary, indent=2))
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="Existing Phoenix HTTP base URL; otherwise start a temporary local server")
    parser.add_argument("--out", type=Path, default=Path("runs/phoenix-smoke"))
    args = parser.parse_args()
    if args.endpoint:
        summary = smoke(args.endpoint, args.out)
    else:
        with local_phoenix(args.out / "server") as endpoint:
            summary = smoke(endpoint, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
