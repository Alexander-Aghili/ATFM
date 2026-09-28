"""Summaries of load-client, control, proxy, and worker observations."""
import json
from collections import Counter
from .metrics import distribution
from .runtime import write_json


def write_report(trial):
    elapsed = trial.loop.time() - trial.origin
    write_json(trial.directory / 'arrivals.json', trial.arrivals)
    write_json(trial.directory / 'tools.json', trial.tools)
    traces = [json.loads(line) for line in (trial.directory / 'proxy-trace.jsonl').read_text().splitlines()]
    successful = [r for r in trial.requests if r['status'] == 200]
    by_request = {(t['session_id'], t['turn_index']): t for t in traces}
    paired = [(r, by_request[(r['session_id'], r['turn'])]) for r in trial.requests
              if (r['session_id'], r['turn']) in by_request]
    metrics = [o for o in trial.observations if 'metrics' in o]
    def maximum(role, section, key):
        return max((o['metrics'][section][key] for o in metrics if o['role'] == role), default=0)
    report = {**_run_metrics(trial, elapsed, traces, successful, paired, maximum),
              **_latency_metrics(trial, elapsed, traces, successful, paired, maximum),
              **_control_metrics(trial, elapsed, traces, successful, paired, maximum)}
    write_json(trial.directory / 'summary.json', report)
    return report


def _run_metrics(trial, elapsed, traces, successful, paired, maximum):
    return {
        'config': trial.cfg.model_dump(),
        'wall_start': trial.wall_origin,
        'elapsed_s': elapsed,
        'workload_elapsed_s': trial.workload_elapsed,
        'cancelled_sessions': trial.cancelled_sessions,
        'proxy_traces_available': len(traces),
        'scheduled_session_rate_s': trial.cfg.sessions / trial.cfg.arrival_window_s,
        'sessions_started': len(trial.arrivals),
        'maximum_requests': trial.cfg.sessions * trial.cfg.turns,
        'requests_attempted': len(trial.requests),
        'requests_ok': len(successful),
        'status_counts': dict(Counter(str(r['status']) for r in trial.requests)),
        'client_errors': dict(Counter(r['error'] for r in trial.requests if r['error'])),
        'drain_deadline_reached': trial.timed_out,
        'session_errors': trial.failures
    }


def _latency_metrics(trial, elapsed, traces, successful, paired, maximum):
    return {
        'completed_request_rate_s': len(successful) / max(trial.workload_elapsed, 1e-9),
        'client_duration_s': distribution(r['duration_s'] for r in trial.requests),
        'client_to_headers_sent_s': distribution(r['transport_s']['http11.send_request_headers.complete']
                                                       for r in trial.requests if 'http11.send_request_headers.complete' in r['transport_s']),
        'client_to_proxy_timestamp_s': distribution(t['t_request'] - r['t'] for r, t in paired),
        'proxy_upstream_s': distribution(t['t_last_token'] - t['t_release'] for t in traces),
        'successful_client_duration_s': distribution(r['duration_s'] for r in successful),
        'proxy_arrival_to_release_s': distribution(t['t_release'] - t['t_request'] for t in traces),
        'session_launch_lateness_s': distribution(r['lateness_s'] for r in trial.arrivals),
        'tool_completion_lateness_s': distribution(r['lateness_s'] for r in trial.tools),
        'control_http_errors': trial.control_client.errors
    }


def _control_metrics(trial, elapsed, traces, successful, paired, maximum):
    return {
        'class_results': {cls: {'requests_ok': sum(r['status'] == 200 for r in trial.requests if r['class'] == cls),
                                      'client_duration_s': distribution(r['duration_s'] for r in trial.requests if r['class'] == cls)}
                                for cls in ('interactive', 'background')},
        'control_steps': len(trial.controls),
        'control_errors': sum(r['errors'] for r in trial.controls),
        'holds_applied': sum(r['holds'] for r in trial.controls),
        'control_step_s': distribution(r['duration_s'] for r in trial.controls),
        'proxy_queue_peak_observed': maximum('proxy', 'queue', 'queued'),
        'worker_waiting_peak_observed': maximum('worker', 'worker', 'waiting'),
        'worker_active_peak': maximum('worker', 'worker', 'max_active'),
        'probe_errors': sum('error' in o for o in trial.observations),
        'tool_publish_errors': trial.bus.errors,
        'final_observations': {role: next((o for o in reversed(trial.observations) if o['role'] == role), None)
                                     for role in trial.endpoints}
    }
