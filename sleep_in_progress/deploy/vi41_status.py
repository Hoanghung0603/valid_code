"""Read-only status for owned vi41 jobs only."""
import json
from pathlib import Path
import subprocess

root=Path('D:/r2ai')
out=root/'data/sub/vi41'
report={}
for p in out.iterdir():
    if p.is_file():
        row={'bytes':p.stat().st_size,'mtime':p.stat().st_mtime}
        if p.suffix=='.log':
            with p.open('rb') as f:
                f.seek(max(0,p.stat().st_size-2000))
                row['tail']=f.read().decode('utf-8',errors='replace')
        elif p.name.endswith('_report.json'):
            row['report']=json.loads(p.read_text(encoding='utf-8'))
        if p.suffix=='.log' or p.name.endswith('_report.json') or p.name=='submission_vi41.zip':
            report[p.name]=row
lock=root/'locks/heavy_job.lock'
report['lock']=lock.read_text() if lock.exists() else None
for name in ('bm25_parts','modal_input','modal_out'):
    directory=out/name
    if directory.exists():
        files=list(directory.iterdir())
        report[name]={'count':len(files),'last':max((p.stat().st_mtime for p in files),default=None)}
report['tasks']={name:subprocess.run(['schtasks','/Query','/TN',name,'/FO','LIST','/V'],
    capture_output=True,text=True).stdout for name in ('r2ai_vi41_qselect','r2ai_vi41_build')}
report['python_processes']=subprocess.run(['powershell','-NoProfile','-Command',
    'Get-Process -Name python -ErrorAction SilentlyContinue'],capture_output=True,text=True).stdout
print(json.dumps(report,ensure_ascii=True,indent=2))
