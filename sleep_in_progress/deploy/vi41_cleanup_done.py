"""After successful build only: remove completed own Task Scheduler definition, never kill."""
import json
from pathlib import Path
import subprocess

root=Path('D:/r2ai')
out=root/'data/sub/vi41'
assert not (root/'locks/heavy_job.lock').exists(),'Owner still active; do not clean'
r=json.loads((out/'build_report.json').read_text())
assert r['queries']==1200 and r['check_zip']==r['validate_exit']==r['verbatim_bad']==0
assert (out/'stats_report.json').exists()
subprocess.run(['schtasks','/Delete','/TN','r2ai_vi41_build','/F'],check=True)
print(json.dumps({'own_build_task_deleted':True,'lock_absent':True,'no_process_killed':True}))
