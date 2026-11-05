"""Read-only final proof: own tasks absent, lock absent, ZIP SHA unchanged."""
import hashlib
import json
from pathlib import Path
import subprocess

root=Path('D:/r2ai')
out=root/'data/sub/vi41'
assert not (root/'locks/heavy_job.lock').exists()
tasks={}
for name in ('r2ai_vi41_build','r2ai_vi41_offline_audit'):
    result=subprocess.run(['schtasks','/Query','/TN',name],capture_output=True,text=True)
    assert result.returncode==1,(name,result.stdout,result.stderr)
    assert 'cannot find' in result.stderr.lower(),result.stderr
    tasks[name]='absent'
zp=out/'submission_vi41.zip'
sha=hashlib.sha256(zp.read_bytes()).hexdigest()
assert sha==json.loads((out/'build_report.json').read_text())['sha256']
report=dict(own_tasks=tasks,lock_absent=True,zip_bytes=zp.stat().st_size,sha256=sha,no_process_killed=True)
(out/'cleanup_cpu_report.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
