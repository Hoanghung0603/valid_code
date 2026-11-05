"""Compact scoped build observation, no mutation."""
import json
from pathlib import Path
import subprocess
O=Path('D:/r2ai/data/sub/vi45')
lock=Path('D:/r2ai/locks/heavy_job.lock')
print('LOCK',lock.read_text() if lock.exists() else 'absent')
for name in ('build.log','submission_vi45.zip','build_report_vi45.json','offline_vi45.json'):
    p=O/name
    if p.exists():
        print(name,p.stat().st_size,p.stat().st_mtime)
        if name=='build.log':
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-1200))
                print(f.read().decode('utf-8','replace'))
print(subprocess.run(['powershell','-NoProfile','-Command',
    'Get-ScheduledTaskInfo -TaskName r2ai_vi45_build'],capture_output=True,text=True).stdout)
