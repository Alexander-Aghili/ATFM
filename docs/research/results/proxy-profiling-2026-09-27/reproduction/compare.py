import json
from pathlib import Path
import subprocess

root = Path('runs/proxy-sniffio-comparison-2026-09-27')
root.mkdir(exist_ok=False)
environments = {'before': '/tmp/atfm-load-check/bin/python', 'after': '/tmp/atfm-proxy-sniffio/bin/python'}
failures = []
for seed in (7, 8, 9):
    for variant in (('before', 'after') if seed % 2 else ('after', 'before')):
        for control in ((True, False) if seed % 2 else (False, True)):
            config = dict(pattern='burst', worker_slots=64, worker_service_s=.005, proxy_window=64,
                          sessions=1024, control_enabled=control, seed=seed)
            path = root / f'{variant}-seed-{seed}-control-{int(control)}'
            code = ('from pathlib import Path; from atfm_experiments.load.__main__ import run_case; '
                    'from atfm_experiments.load.config import LoadConfig; '
                    f'r=run_case(LoadConfig(**{config!r}),Path({str(path)!r})); '
                    'print(r["requests_ok"],r["client_duration_s"]["p95"],flush=True); '
                    'assert r["requests_ok"]==r["maximum_requests"] and not any(r[k] for k in '
                    '("session_errors","drain_deadline_reached","tool_publish_errors","control_errors",'
                    '"control_http_errors","probe_errors"))')
            print(path.name, flush=True)
            result = subprocess.run([environments[variant], '-c', code])
            if result.returncode:
                failures.append(path.name)
(root / 'failures.json').write_text(json.dumps(failures) + '\n')
raise SystemExit(bool(failures))
