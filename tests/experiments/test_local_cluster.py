"""Bounded trace selection preserves original requests and subagent structure."""
import hashlib
import json

import pytest

from atfm_experiments.local_cluster.workload import requests, select_trace


def test_selects_complete_smallest_root_including_children(tmp_path):
    child = dict(type='n', **{'in': 100, 'out': 1})
    tiny = dict(type='s', **{'in': 2, 'out': 1})
    large = dict(id='large', requests=[tiny, dict(type='subagent', requests=[child])])
    small = dict(id='small', requests=[tiny, tiny])
    source = tmp_path / 'source.jsonl'
    source.write_text('\n'.join(json.dumps(t) for t in [large, small]))
    report = select_trace(source, tmp_path)
    assert json.loads((tmp_path / 'trace.json').read_text()) == small
    assert report['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert report['roots_scanned'] == 2 and report['requests'] == 2
    assert len(list(requests(large))) == 2


def test_empty_corpus_cannot_produce_success(tmp_path):
    source = tmp_path / 'empty.jsonl'
    source.write_text('\n')
    with pytest.raises(ValueError, match='no inference'):
        select_trace(source, tmp_path)


@pytest.mark.parametrize('change', [dict(is_complete=False), dict(was_cancelled=True),
                                  dict(error_summary=[{'error': 'timeout'}]),
                                  dict(branch_stats={'children_truncated': 1}),
                                  dict(request_count={'avg': 0})])
def test_partial_or_failed_replay_is_not_success(tmp_path, change):
    from atfm_experiments.local_cluster.__main__ import inspect_result
    report = dict(is_complete=True, was_cancelled=False, error_summary=[], request_count={'avg': 20}, branch_stats={})
    (tmp_path / 'aiperf').mkdir()
    (tmp_path / 'aiperf/profile_export_aiperf.json').write_text(json.dumps(report | change))
    with pytest.raises(ValueError):
        inspect_result(tmp_path)


def test_valid_replay_keeps_evidence_scope_explicit(tmp_path):
    from atfm_experiments.local_cluster.__main__ import inspect_result
    report = dict(is_complete=True, was_cancelled=False, error_summary=[], request_count={'avg': 20}, branch_stats={})
    (tmp_path / 'aiperf').mkdir()
    (tmp_path / 'aiperf/profile_export_aiperf.json').write_text(json.dumps(report))
    result = inspect_result(tmp_path)
    assert result['status'] == 'passed' and result['requests'] == 20
    assert not any(result[k] for k in ('hardware_evidence', 'lmcache_tested', 'atfm_control_tested'))


def test_client_timeout_terminates_and_reaps_process_group(tmp_path, monkeypatch):
    import os
    import subprocess
    import sys
    from unittest.mock import Mock
    from atfm_experiments.local_cluster.process import run_client
    signal_group = Mock(wraps=os.killpg)
    monkeypatch.setattr('atfm_experiments.local_cluster.process.os.killpg', signal_group)
    with (tmp_path / 'client.log').open('w') as log:
        with pytest.raises(subprocess.TimeoutExpired):
            run_client([sys.executable, '-c', 'import time; time.sleep(30)'], log, timeout=.05)
    pid = signal_group.call_args.args[0]
    with pytest.raises(ProcessLookupError):
        os.killpg(pid, 0)
