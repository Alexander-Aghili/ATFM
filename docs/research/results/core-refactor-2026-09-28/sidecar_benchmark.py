import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

from atfm.bus import InMemoryBus
from atfm.sidecar.core import ToolContext, run_tool
from atfm.sidecar.adapters import wrap_executor
from atfm.sidecar.minisweagent import SidecarConfig

payload = ''.join(f'processed {i}/10000 rows\n' for i in range(10000))
def trial(kind):
    bus = InMemoryBus()
    start = time.perf_counter()
    if kind == 'wrapped':
        result = {'output': payload, 'returncode': 0}
        fn = wrap_executor(lambda cmd: result, SidecarConfig(session_id='s', cls='interactive', bus=bus, clock=lambda: 10.))
        assert fn('tool') is result
        output = result['output'].encode()
    else:
        result = run_tool([sys.executable, '-c', "import sys;sys.stdout.write(''.join(f'processed {i}/10000 rows\\n' for i in range(10000)))"],
                          ToolContext('s', 0, 'tool'), bus, clock=lambda: 10.)
        output = result.output
        assert result.returncode == 0
    elapsed = time.perf_counter() - start
    events = [e.model_dump(exclude={'call_id'}) for e in bus.drain()]
    return elapsed, hashlib.sha256(output + json.dumps(events,sort_keys=True).encode()).hexdigest(), len(events)

out={}
for kind in ('wrapped','subprocess'):
    trial(kind)
    rows=[trial(kind) for _ in range(7)]
    assert len({r[1] for r in rows})==1
    out[kind]={'seconds':[r[0] for r in rows], 'median_s':statistics.median(r[0] for r in rows),
               'sha256':rows[0][1], 'events':rows[0][2]}
Path(sys.argv[1]).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
