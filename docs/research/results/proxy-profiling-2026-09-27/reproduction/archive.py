from pathlib import Path
import csv
import gzip
import hashlib
import json
import shutil

root = Path('docs/research/results/proxy-profiling-2026-09-27')
groups = {
    'exploratory-cprofile': 'runs/proxy-profile-2026-09-27',
    'cpu-before': 'runs/proxy-cpu-profile-2026-09-27',
    'cpu-after': 'runs/proxy-cpu-profile-sniffio-2026-09-27',
    'unprofiled': 'runs/proxy-sniffio-comparison-2026-09-27',
}
manifest, rows = {}, []
def retain(source, destination, compress=False):
    raw = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if compress:
        destination = destination.with_name(destination.name + '.gz')
        destination.write_bytes(gzip.compress(raw, mtime=0))
    else:
        shutil.copyfile(source, destination)
    manifest[str(destination.relative_to(root))] = {
        'raw_bytes': len(raw), 'raw_sha256': hashlib.sha256(raw).hexdigest()}

for group, source in groups.items():
    for summary in sorted(Path(source).glob('*/summary.json')):
        case = summary.parent
        for path in sorted(case.iterdir()):
            if path.is_file():
                retain(path, root / group / case.name / path.name,
                       path.name not in ('config.json', 'environment.json', 'summary.json', 'shutdown.json',
                                         'proxy-profile.csv', 'proxy-profile.json'))
        r = json.loads(summary.read_text())
        obs = r['final_observations']
        p = obs['proxy']['metrics']['predictions']
        rows.append(dict(group=group, case=case.name, seed=r['config']['seed'],
                         control=r['config']['control_enabled'], profiled=r['config']['profile_proxy'],
                         requests_ok=r['requests_ok'], maximum_requests=r['maximum_requests'],
                         p95_s=r['client_duration_s']['p95'], throughput=r['completed_request_rate_s'],
                         proxy_cpu_s=obs['proxy']['metrics']['cpu_s'],
                         prediction_attempted=p['attempted'], prediction_used=p['used'],
                         prediction_timeout=p['timeout'], prediction_error=p['error'],
                         transport_p95_s=r['client_to_headers_sent_s']['p95'],
                         proxy_loop_lag_max_s=obs['proxy']['metrics']['event_loop_lag_s']['max'],
                         control_errors=r['control_errors'], control_http_errors=r['control_http_errors'],
                         probe_errors=r['probe_errors'], tool_errors=r['tool_publish_errors'],
                         deadline=r['drain_deadline_reached']))
for name in ('detection.py','detection-before.json','detection-after.json','compare.py',
             'cprofile_threads.py','cprofile-thread-result.json','packages-before.json','packages-after.json',
             'config.json','yappi.json','archive.py'):
    retain(Path('tmp/proxy-profile') / name, root / 'reproduction' / name)
(root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
with (root / 'comparison.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
for name, entry in manifest.items():
    path = root / name
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    assert len(raw) == entry['raw_bytes']
    assert hashlib.sha256(raw).hexdigest() == entry['raw_sha256']
print('Verified', len(manifest), 'files;', len(rows), 'cases')
