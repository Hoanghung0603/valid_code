"""Read-only b660 status, scoped only to vi45 and shared lock."""
import json
from pathlib import Path
import subprocess

ROOT=Path('D:/r2ai')
OUT=ROOT/'data/sub/vi45'
lock=ROOT/'locks/heavy_job.lock'
print('LOCK',lock.read_text(encoding='utf-8') if lock.exists() else 'absent')
for stage in ('supplement','preverify','build'):
    p=OUT/(stage+'.log')
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-1800))
            print(stage,p.stat().st_size,p.stat().st_mtime,f.read().decode('utf-8','replace'))
for name in ('prepare_vi44_report.json','preverify_report.json','build_report_vi45.json','offline_vi45.json'):
    p=OUT/name
    if p.exists():
        r=json.loads(p.read_text(encoding='utf-8'))
        print(name,json.dumps({k:v for k,v in r.items() if k not in ('files','counts','per_query','input_fingerprints')}))
for task in ('r2ai_vi45_supplement','r2ai_vi45_preverify','r2ai_vi45_build'):
    p=subprocess.run(['schtasks','/Query','/TN',task,'/FO','LIST','/V'],capture_output=True,text=True)
    print('TASK',task,p.returncode,p.stdout)
print(subprocess.run(['powershell','-NoProfile','-Command',
    "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" -Property ProcessId,CommandLine"],capture_output=True,text=True).stdout)
