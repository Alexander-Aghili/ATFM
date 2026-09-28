"""Run the control loop against a board service and a proxy.

    uv run python scripts/run_control.py --board http://127.0.0.1:8081 --proxy http://127.0.0.1:8799 --interval 5
"""
import argparse

from atfm.control.loop import ControlLoop


def main():
    a = _arguments()
    lm = None
    if a.lmcache:
        import httpx
        from transformers import AutoTokenizer
        from atfm.control.lmcache import LMCacheActuator, LMCacheConfig, PromptTokens
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
        http = httpx.Client(timeout=2.0)

        def prompt_source(sid):
            r = http.get(f"{a.proxy.rstrip('/')}/session/{sid}/prompt")
            return r.json().get("messages") if r.status_code == 200 else None

        def tokenize(messages):
            return list(tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
        lm = LMCacheActuator(LMCacheConfig(url=a.lmcache, instance_id=a.lmcache_instance), tokens=PromptTokens(tokenize, prompt_source))
    ControlLoop(a.board, a.proxy, interval_s=a.interval, log_path=a.log, lmcache=lm).run(steps=a.steps)


def _arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument("--board", default="http://127.0.0.1:8081")
    ap.add_argument("--proxy", default="http://127.0.0.1:8799")
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--log", default="runs/control/control.jsonl")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lmcache", default=None, help="LMCache controller URL: execute placement by pin/move instead of touches")
    ap.add_argument("--lmcache-instance", default="vllm-0")
    ap.add_argument("--tokenizer", default=None, help="HF tokenizer name for prompt token ids (required with --lmcache)")
    a = ap.parse_args()
    return a


if __name__ == "__main__":
    main()
