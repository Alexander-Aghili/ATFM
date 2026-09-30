"""Output length comes from the OpenAI field the client actually sent."""
from atfm.proxy.app import requested_osl


def test_prefers_max_completion_tokens_then_max_tokens_then_default():
    assert requested_osl({"max_completion_tokens": 587, "max_tokens": 9}, 256) == 587
    assert requested_osl({"max_tokens": 9}, 256) == 9
    assert requested_osl({}, 256) == 256


def test_malformed_lengths_fall_back_to_the_default():
    assert requested_osl({"max_completion_tokens": "many"}, 256) == 256
    assert requested_osl({"max_tokens": None}, 256) == 256
