"""Stop ONLY own vi41 command-line workers after repeated Modal import failure."""
import argparse
import json
from pathlib import Path
import subprocess

ap=argparse.ArgumentParser()
ap.add_argument('--machine',choices=['b660','z87'],required=True)
a=ap.parse_args()
root=Path('D:/r2ai')
dest=root/'data/sub/vi41'
command=['powershell','-NoProfile','-Command',
         "ConvertTo-Json -InputObject @(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\") -Depth 3"]
result=subprocess.run(command,capture_output=True,text=True,check=True)
rows=json.loads(result.stdout)
killed=[]
for row in rows:
    cmd=row.get('CommandLine') or ''
    if 'scripts\\vi41.py' in cmd or 'scripts/vi41.py' in cmd:
        pid=int(row['ProcessId'])
        r=subprocess.run(['taskkill','/PID',str(pid),'/T','/F'],capture_output=True,text=True)
        killed.append(dict(pid=pid,command=cmd,exit=r.returncode,stdout=r.stdout,stderr=r.stderr))
tasks=['r2ai_vi41_neardup'] if a.machine=='b660' else ['r2ai_vi41_dense']
deleted=[]
for task in tasks:
    r=subprocess.run(['schtasks','/Delete','/TN',task,'/F'],capture_output=True,text=True)
    deleted.append(dict(task=task,exit=r.returncode,stdout=r.stdout,stderr=r.stderr))
lock=root/'locks/heavy_job.lock'
before=lock.read_text(encoding='utf-8') if lock.exists() else None
if before and before.startswith('vi41 ') and any('pid='+str(r['pid']) in before for r in killed):
    assert all(r['exit']==0 for r in killed)
    lock.unlink()
report=dict(machine=a.machine,killed=killed,tasks=deleted,lock_before=before,
            lock_after=lock.read_text(encoding='utf-8') if lock.exists() else None,
            reason="Repeated Modal container import ModuleNotFoundError: No module named 'modal_dchunk'; STOP, no respawn")
(dest/'cleanup_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,ensure_ascii=True))
