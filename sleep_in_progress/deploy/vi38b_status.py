"""Read-only status of our vi38b tasks, logs, lock, and final artefacts."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('D:/r2ai')
base = root/'data/sub/vi38'
lock = root/'locks/heavy_job.lock'
report = dict(lock=lock.read_text(encoding='utf-8') if lock.exists() else None)
for mode in ('prepare', 'finish', 'audit'):
    ob = subprocess.run(['schtasks', '/Query', '/TN', 'r2ai_vi38b_'+mode, '/FO', 'LIST', '/V'],
                        capture_output=True, text=True)
    report['task_'+mode] = dict(exit=ob.returncode, output=ob.stdout)
    log = base/f'vi38b_{mode}.log'
    if log.exists():
        lines = log.read_text(encoding='utf-8', errors='replace').splitlines()
        report['log_'+mode] = dict(mtime=log.stat().st_mtime,
                                       tail='\n'.join(line[:800] for line in lines[-8:]))
for name in ('merge_report.json', 'confirm_report.json', 'cmerge_report.json',
             'build_report_vi38b.json', 'stats_report_vi38b.json', 'gate2_report.json'):
    p = base/name
    if p.exists():
        ob = json.loads(p.read_text(encoding='utf-8'))
        report[name] = {k: v for k, v in ob.items() if k not in (
            'jaccard_by_query', 'rank_distribution', 'query_counts', 'files',
            'eligible_by_query', 'eligible_after_by_query')}
audit = base/'independent_audit_vi38b.json'
if audit.exists():
    report[audit.name] = json.loads(audit.read_text(encoding='utf-8'))
for name in ('submission_vi38b.zip', 'gate2_package.zip', 'confirm_input.zip'):
    p = base/name
    if p.exists() and (name != 'submission_vi38b.zip' or (base/'build_report_vi38b.json').exists()):
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(4*1024*1024), b''):
                h.update(block)
        report[name] = dict(bytes=p.stat().st_size, sha256=h.hexdigest())
(base/'vi38b_status.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
