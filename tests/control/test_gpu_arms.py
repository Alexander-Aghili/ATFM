from pathlib import Path

import pytest

from atfm_experiments.gpu_cache import arms, replay
from atfm_experiments.gpu_cache.stack import CACHE, INFERENCE


def names(arm, tmp_path):
    return [name for name, _, _ in arms.processes(arm, Path('/venv/python'), tmp_path, Path('train.parquet'), 24, 1.4)]


def test_direct_arm_adds_no_processes_and_targets_vllm(tmp_path):
    assert names('direct', tmp_path) == []
    assert arms.client_url('direct') == INFERENCE and arms.client_flags('direct') == []


def test_proxy_arm_puts_only_the_proxy_in_the_path(tmp_path):
    ((name, command, ready),) = arms.processes('proxy', Path('/venv/python'), tmp_path, Path('t.parquet'), 24, 1.4)
    assert name == 'proxy' and ready == arms.PROXY + '/healthz' and '--board' not in command
    assert command[command.index('--upstream') + 1] == INFERENCE
    assert arms.client_url('proxy') == arms.PROXY
    flags = arms.client_flags('proxy')
    assert flags[flags.index('--session-header') + 1] == 'x-atfm-session'
    assert flags[flags.index('--server-metrics') + 1] == INFERENCE + '/metrics'


def test_atfm_arm_starts_board_then_proxy_then_control_with_async_lmcache(tmp_path):
    procs = arms.processes('atfm', Path('/venv/python'), tmp_path, Path('train.parquet'), 24, 1.4)
    assert [n for n, _, _ in procs] == ['board', 'proxy', 'control']
    board, proxy, control = (c for _, c, _ in procs)
    assert '--gap-after-done' in board and board[board.index('--train') + 1] == 'train.parquet'
    assert proxy[proxy.index('--board') + 1] == arms.BOARD
    assert control[control.index('--lmcache') + 1] == CACHE and '--lmcache-async' in control
    assert control[control.index('--lmcache-chunk-size') + 1] == '16'
    assert (tmp_path / 'control.yaml').exists() and procs[2][2] is None


def test_control_config_budgets_half_the_cpu_tier_and_uses_the_calibrated_rate():
    cfg = arms.control_config(l1_gb=24, warm_gbps=1.4, interval_s=2.0)['prefetch']
    assert cfg['budget_bytes'] == 12 * 2**30 and cfg['warm_bytes_per_s'] == 1.4e9
    assert cfg['bytes_per_token'] == 147456 and cfg['interval_s'] == 2.0


def test_unknown_arm_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        arms.processes('oracle', Path('/p'), tmp_path, Path('t'), 24, 1.4)


def test_replay_command_targets_the_arm_url_with_its_flags(tmp_path):
    cmd = replay.command(tmp_path / 'aiperf', tmp_path / 'trace', tmp_path, url=arms.PROXY, extra=['--x', 'y'])
    assert cmd[cmd.index('--url') + 1] == arms.PROXY and cmd[-2:] == ['--x', 'y']


def test_replay_cli_takes_arm_training_table_and_warm_rate():
    args = replay.parse(['--source', 's', '--output', 'o', '--arm', 'atfm', '--train', 't.parquet', '--warm-gbps', '1.6'])
    assert (args.arm, args.train, args.warm_gbps) == ('atfm', Path('t.parquet'), 1.6)
    assert replay.parse(['--source', 's', '--output', 'o']).arm == 'direct'


def test_owned_process_without_readiness_url_starts_and_stops(tmp_path):
    from atfm_experiments.gpu_cache.stack import server
    with server(['sleep', '30'], tmp_path, 'sleeper', None) as process:
        assert process.poll() is None
    assert process.poll() is not None


def test_prefetch_policy_reaches_the_board_config(tmp_path):
    import yaml
    arms.processes('atfm', Path('/p'), tmp_path, Path('t'), 24, 1.4, policy=dict(trigger='q50', rewarm_after_s=30.0))
    cfg = yaml.safe_load((tmp_path / 'control.yaml').read_text())['prefetch']
    assert (cfg['trigger'], cfg['rewarm_after_s']) == ('q50', 30.0)
    args = replay.parse(['--source', 's', '--output', 'o', '--arm', 'atfm', '--train', 't',
                         '--prefetch-trigger', 'q50', '--rewarm-after', '30'])
    assert (args.prefetch_trigger, args.rewarm_after) == ('q50', 30.0)
