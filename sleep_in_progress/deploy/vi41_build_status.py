"""Concise read-only vi41 build status; no unrelated task/process details."""
import json
from pathlib import Path
import subprocess

root=Path('D:/r2ai')
out=root/'data/sub/vi41'
report={}
for name in ('merge','chunk','build','stats','offline_extra'):
    log=out/(name+'.log')
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-2500))
            tail=f.read().decode('utf-8',errors='replace')
        report[name]={'mtime':log.stat().st_mtime,'tail':tail}
for name in ('chunk_report.json','build_report.json','stats_report.json','offline_extra_report.json'):
    p=out/name
    if p.exists():
        report[name]=json.loads(p.read_text(encoding='utf-8'))
for name in ('submission_vi41.json','submission_vi41.zip'):
    p=out/name
    if p.exists():
        report[name]={'bytes':p.stat().st_size,'mtime':p.stat().st_mtime}
lock=root/'locks/heavy_job.lock'
report['lock']=lock.read_text() if lock.exists() else None
report['task']=subprocess.run(['schtasks','/Query','/TN','r2ai_vi41_build','/FO','LIST','/V'],capture_output=True,text=True).stdout
if lock.exists() and 'pid=' in report['lock']:
    pid=int(report['lock'].rsplit('pid=',1)[1].strip())
    report['owner_process']=subprocess.run(['powershell','-NoProfile','-Command',f'Get-Process -Id {pid} -ErrorAction SilentlyContinue'],capture_output=True,text=True).stdout
print(json.dumps(report))
