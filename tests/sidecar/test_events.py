from types import SimpleNamespace

from atfm.bus import InMemoryBus
from atfm.sidecar.events import ProgressEmitter


def test_parser_order_data_fallthrough_and_duplicate_progress():
    bus = InMemoryBus()
    visited = []
    data = SimpleNamespace(feed=lambda *_: None, feed_data=lambda *_: {'metric': 'loss', 'value': .5})
    progress = SimpleNamespace(feed=lambda *_: {'completed': 1, 'total': 3})
    tail = SimpleNamespace(feed=lambda *_: visited.append('tail'))
    emitter = ProgressEmitter(bus, 's', 'call', [data, progress, tail])
    emitter.feed('first', 1)
    emitter.feed('duplicate', 2)
    events = bus.drain()
    assert [e.kind for e in events] == ['tool.data', 'tool.progress', 'tool.data']
    assert [e.t for e in events] == [1, 1, 2]
    assert not visited


def test_parser_and_bus_errors_preserve_fallback_and_deduplication():
    bus = InMemoryBus()
    broken = SimpleNamespace(feed=lambda *_: {'completed': 'bad'})
    fallback = SimpleNamespace(feed=lambda *_: {'completed': 2, 'total': 4})
    emitter = ProgressEmitter(bus, 's', 'call', [broken, fallback])
    emitter.feed('invalid progress', 1)
    assert bus.drain()[0].completed == 2

    attempts = []
    def fail(event):
        attempts.append(event)
        raise RuntimeError('bus unavailable')

    emitter = ProgressEmitter(SimpleNamespace(publish=fail), 's', 'call', [fallback])
    emitter.feed('first', 1)
    emitter.feed('duplicate', 2)
    assert len(attempts) == 1


def test_configuration_compatibility_import_and_adapter_independence():
    import subprocess
    import sys
    from atfm.sidecar.config import SidecarConfig
    from atfm.sidecar.minisweagent import SidecarConfig as LegacyConfig

    assert SidecarConfig is LegacyConfig
    result = subprocess.run([sys.executable, '-c',
                             "import sys; import atfm.sidecar.adapters; "
                             "assert 'atfm.sidecar.minisweagent' not in sys.modules; "
                             "assert 'minisweagent' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
