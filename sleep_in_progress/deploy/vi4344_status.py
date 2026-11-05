"""Read only own CPU tasks, own logs, and shared lock; no crawl access."""
import json
from pathlib import Path
import subprocess

root = Path('D:/r2ai')
lock = root/'locks/heavy_job.lock'
print('LOCK', lock.read_text(encoding='utf-8') if lock.exists() else 'absent')
for name in ('vi44', 'vi43'):
    out = root/'data/sub'/name
    for p in sorted(out.glob('*')):
        if p.is_file():
            print(name, p.name, p.stat().st_size, p.stat().st_mtime)
    log = out/f'{name}.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0, log.stat().st_size-2500))
            print(f.read().decode('utf-8', 'replace'))
    subprocess.run(['schtasks', '/Query', '/TN', f'r2ai_{name}', '/V', '/FO', 'LIST'])
subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe', '/FO', 'CSV'])
