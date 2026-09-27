"""Parser chain additions (spec 7): dbt/Spark rows and stages, training monitors."""
from atfm.sidecar.parsers import RowsParser, StageParser, TrainingParser, default_parsers


def test_rows_parser_reads_k_of_n_and_rows_forms():
    p = RowsParser()
    assert p.feed("rows processed 120/600", 1.0) == {"completed": 120.0, "total": 600.0, "phase": "rows"}
    assert p.feed("processed 300 of 600 rows", 2.0) == {"completed": 300.0, "total": 600.0, "phase": "rows"}
    assert p.feed("[INFO] 450/600 rows written", 3.0) == {"completed": 450.0, "total": 600.0, "phase": "rows"}
    assert p.feed("no numbers here", 4.0) is None and p.feed("done 5/3 rows", 5.0) is None


def test_stage_parser_reads_stage_markers():
    p = StageParser()
    assert p.feed("stage 2/4: transform", 1.0) == {"completed": 2.0, "total": 4.0, "phase": "stage"}
    assert p.feed("Stage 3 of 4", 2.0) == {"completed": 3.0, "total": 4.0, "phase": "stage"}
    assert p.feed("[stage 4/4]", 3.0) == {"completed": 4.0, "total": 4.0, "phase": "stage"}
    assert p.feed("stagecoach 1/2", 4.0) is None


def test_training_parser_progress_and_loss_data():
    p = TrainingParser()
    assert p.feed("step 120/1000 loss=0.512 lr=1e-4", 1.0) == {"completed": 120.0, "total": 1000.0, "phase": "step"}
    assert p.feed_data("step 120/1000 loss=0.512 lr=1e-4", 1.0) == {"metric": "loss", "value": 0.512}
    assert p.feed("epoch 2/10", 2.0) == {"completed": 2.0, "total": 10.0, "phase": "epoch"}
    assert p.feed_data("train loss: 0.25", 3.0) == {"metric": "loss", "value": 0.25}
    assert p.feed("nothing", 4.0) is None and p.feed_data("nothing", 4.0) is None


def test_default_chain_order_puts_specific_parsers_before_the_generic_counter():
    names = [type(p).__name__ for p in default_parsers()]
    assert names.index("RowsParser") < names.index("CounterParser") and names.index("StageParser") < names.index("CounterParser")
    assert names.index("TrainingParser") < names.index("CounterParser") and names[0] == "PytestParser" and names[-1] == "RateParser"
