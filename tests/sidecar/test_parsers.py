from atfm.sidecar.parsers import PytestParser, PercentParser, CounterParser, RateParser
from atfm.sidecar.core import classify_tool

def test_pytest_parser_counts_results():
    p = PytestParser()
    assert p.feed("collected 3 items", 0.0) == {"completed": 0, "total": 3, "phase": "collect"}
    assert p.feed("tests/test_a.py::test_x PASSED                    [ 33%]", 1.0) == {"completed": 1, "total": 3, "phase": "run"}
    assert p.feed("tests/test_a.py::test_y FAILED                    [ 66%]", 2.0) == {"completed": 2, "total": 3, "phase": "run"}
    assert p.feed("some unrelated line", 2.5) is None
    assert p.feed("=========== 1 failed, 1 passed in 2.10s ===========", 3.0) == {"completed": 3, "total": 3, "phase": "done"}

def test_percent_and_counter_parsers():
    assert PercentParser().feed("[ 45%] Building C object x.c.o", 0.0) == {"completed": 45.0, "total": 100.0, "phase": "run"}
    assert CounterParser().feed("[12/345] compiling foo", 0.0) == {"completed": 12.0, "total": 345.0, "phase": "run"}
    assert CounterParser().feed("processed 10/100 rows", 0.0) == {"completed": 10.0, "total": 100.0, "phase": "run"}
    assert CounterParser().feed("version 1/2", 0.0) is None
    assert CounterParser().feed("k 500/100", 0.0) is None

def test_rate_parser_emits_every_window():
    r = RateParser(window_s=5.0)
    assert r.feed_data("a", 0.0) is None
    for i in range(9):
        assert r.feed_data("x", 1.0 + i * 0.5) is None
    d = r.feed_data("y", 5.5)
    assert d["metric"] == "lines_per_s" and abs(d["value"] - 11 / 5.5) < 1e-9

def test_classify_tool():
    assert classify_tool("cd /w && pytest -q tests") == "pytest"
    assert classify_tool("cmake --build build -j4") == "build"
    assert classify_tool("pip install -e .") == "install"
    assert classify_tool("git clone https://x/y") == "clone"
    assert classify_tool("cd /w && ls -la") == "ls"
