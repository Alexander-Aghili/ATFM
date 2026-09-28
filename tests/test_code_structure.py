"""Keep the function-size convention enforceable across maintained Python code."""
from scripts.check_function_size import tracked_sources, violations


def test_tracked_functions_fit_twenty_lines():
    assert violations(tracked_sources()) == []


def test_size_check_includes_async_nested_functions_and_docstrings(tmp_path):
    source = tmp_path / 'example.py'
    source.write_text('async def outer():\n    def inner():\n        """' + '\n' * 19 + '        """\n')
    failures = violations([source])
    assert len(failures) == 2
    assert any('outer' in failure for failure in failures)
    assert any('inner' in failure for failure in failures)
