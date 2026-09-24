import os, pytest
pytestmark = pytest.mark.skipif(os.environ.get("ATFM_DYNAMO") != "1", reason="set ATFM_DYNAMO=1 with the dynamo extra installed")

def test_mocker_frontend_serves_hinted_request(tmp_path):
    from atfm.dynamo.local import LocalDynamo
    with LocalDynamo(port=8790, log_dir=str(tmp_path)) as d:
        r = d.chat([{"role": "user", "content": "hello"}], max_tokens=8,
                   hints={"priority": 3, "strict_priority": 1, "osl": 8}, session_id="smoke-1")
        assert "choices" in r and r["choices"][0]["message"]["content"]
