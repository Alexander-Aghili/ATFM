from types import SimpleNamespace

from atfm.bus import InMemoryBus
from atfm.sidecar.adapters import wrap_executor
from atfm.sidecar.config import SidecarConfig


def feed_lines(bus, parsers, lines, monkeypatch):
    monkeypatch.setattr('atfm.sidecar.adapters.default_parsers', lambda: parsers)
    result = {'output': "\n".join(lines), 'returncode': 0}
    wrapped = wrap_executor(lambda _: result, SidecarConfig(session_id='s', cls='interactive',
                                                           bus=bus, clock=lambda: 2.))
    assert wrapped('tool') is result


def test_parser_order_data_fallthrough_and_duplicate_progress(monkeypatch):
    bus = InMemoryBus()
    visited = []
    data = SimpleNamespace(feed=lambda *_: None, feed_data=lambda *_: {'metric': 'loss', 'value': .5})
    progress = SimpleNamespace(feed=lambda *_: {'completed': 1, 'total': 3})
    tail = SimpleNamespace(feed=lambda *_: visited.append('tail'))
    feed_lines(bus, [data, progress, tail], ['first', 'duplicate'], monkeypatch)
    events = [e for e in bus.drain() if e.kind in ("tool.data", "tool.progress")]
    assert [e.kind for e in events] == ['tool.data', 'tool.progress', 'tool.data']
    assert [e.t for e in events] == [2, 2, 2]
    assert not visited


def test_parser_and_bus_errors_preserve_fallback_and_deduplication(monkeypatch):
    bus = InMemoryBus()
    broken = SimpleNamespace(feed=lambda *_: {'completed': 'bad'})
    fallback = SimpleNamespace(feed=lambda *_: {'completed': 2, 'total': 4})
    feed_lines(bus, [broken, fallback], ['invalid progress'], monkeypatch)
    assert next(e for e in bus.drain() if e.kind == "tool.progress").completed == 2

    attempts = []
    def fail(event):
        attempts.append(event)
        raise RuntimeError('bus unavailable')

    feed_lines(SimpleNamespace(publish=fail), [fallback], ['first', 'duplicate'], monkeypatch)
    assert sum(e.kind == "tool.progress" for e in attempts) == 1


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


def test_mixin_uses_explicit_builder_even_if_it_is_falsey():
    import sys
    from atfm.sidecar.minisweagent import SidecarConfig, SidecarMixin

    class Builder:
        def __bool__(self):
            return False

        def __call__(self, command):
            return [sys.executable, '-c', "print('from builder')"]

    class Environment(SidecarMixin):
        def _check_finished(self, output):
            self.checked = output

    env = Environment(sidecar=SidecarConfig(session_id='s', cls='interactive'))
    result = env.sidecar_execute('not a shell command', '', 5, Builder())
    assert result['output'] == 'from builder\n'
    assert env.checked is result
