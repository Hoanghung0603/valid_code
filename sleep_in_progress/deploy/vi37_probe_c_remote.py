"""Small read-only liveness probe limited to vi37 task/files/processes."""
from datetime import datetime
import json
from pathlib import Path
import subprocess

OUT = Path('D:/r2ai/data/sub/vi37')
result = {'now': datetime.now().isoformat(), 'files': {}}
for name in ['vi37.log', 'submission_vi37p.zip', 'submission_vi37.zip', 'attempts_vi37p.json']:
    p = OUT / name
    if p.exists():
        st = p.stat()
        result['files'][name] = {'bytes': st.st_size, 'mtime': datetime.fromtimestamp(st.st_mtime).isoformat()}
        if name == 'attempts_vi37p.json':
            result['attempts'] = {k: {kk: vv for kk, vv in v.items() if kk != 'hist_kcap'}
                                  for k, v in json.loads(p.read_text(encoding='utf-8')).items()}
cmd = "Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | Where-Object {$_.CommandLine -match 'vi37|zip_submission'} | Select-Object ProcessId,ParentProcessId,CommandLine,KernelModeTime,UserModeTime | ConvertTo-Json"
p = subprocess.run(['powershell', '-NoProfile', '-Command', cmd], capture_output=True, text=True)
result['processes'] = p.stdout.strip() or p.stderr.strip()
print(json.dumps(result, ensure_ascii=False))
