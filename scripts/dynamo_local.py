import json
import os
import signal
import sys
from pathlib import Path

from atfm.dynamo.local import LocalDynamo

PIDS = Path("runs/dynamo/pids.json")


def main(cmd: str) -> None:
    if cmd == "up":
        d = LocalDynamo()
        d.start()
        PIDS.write_text(json.dumps([p.pid for p in d.procs]))
        print(f"up at {d.base_url}; pids {PIDS.read_text()}")
    elif cmd == "down":
        for pid in json.loads(PIDS.read_text()) if PIDS.exists() else []:
            try:
                os.killpg(pid, signal.SIGTERM)
            except Exception:
                pass
        print("down")
    elif cmd == "smoke":
        with LocalDynamo(port=8790) as d:
            r = d.chat([{"role": "user", "content": "hello"}], hints={"priority": 3, "strict_priority": 1, "osl": 8},
                       session_id="smoke")
            print(json.dumps(r, indent=1)[:800])
            print((d.log_dir / "frontend.log").read_text()[-1500:])
    else:
        raise SystemExit("usage: dynamo_local.py up|down|smoke")


if __name__ == "__main__":
    main(sys.argv[1])
