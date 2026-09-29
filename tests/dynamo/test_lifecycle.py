"""Subprocess ownership must survive partial startup and cancellation."""
from unittest.mock import Mock

import pytest

from atfm.dynamo.local import LocalDynamo


@pytest.mark.parametrize('error', [RuntimeError('launch'), KeyboardInterrupt()])
def test_start_cleans_partial_launch(tmp_path, monkeypatch, error):
    cluster = LocalDynamo(log_dir=tmp_path)
    monkeypatch.setattr(cluster, '_launch', Mock(side_effect=error))
    cleanup = Mock()
    monkeypatch.setattr(cluster, 'stop', cleanup)
    with pytest.raises(type(error)):
        cluster.start()
    cleanup.assert_called_once()


def test_spawn_closes_parent_log_handle(tmp_path, monkeypatch):
    spawn = Mock(return_value=Mock())
    monkeypatch.setattr('atfm.dynamo.local.subprocess.Popen', spawn)
    cluster = LocalDynamo(log_dir=tmp_path)
    child = cluster._spawn('worker', ['dynamo.mocker'])
    assert spawn.call_args.kwargs['stdout'].closed
    assert cluster.procs == [child]


def test_stop_reaps_children(tmp_path, monkeypatch):
    child = Mock(pid=123, poll=Mock(return_value=0))
    monkeypatch.setattr('atfm.dynamo.local.os.killpg', Mock())
    cluster = LocalDynamo(log_dir=tmp_path)
    cluster.procs = [child]
    cluster.stop()
    child.wait.assert_called_once()
    assert cluster.procs == []
