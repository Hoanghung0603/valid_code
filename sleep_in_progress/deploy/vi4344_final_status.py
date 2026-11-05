"""Verify only this run's tasks/processes/lock; read-only, no crawl access."""
from pathlib import Path
import subprocess

lock = Path('D:/r2ai/locks/heavy_job.lock')
owner = lock.read_text(encoding='utf-8') if lock.exists() else 'absent'
assert not owner.startswith(('vi44 ', 'vi43 ')), owner
print('LOCK', owner, '(foreign owner left untouched)' if owner != 'absent' else '', flush=True)
for task in ('r2ai_vi44', 'r2ai_vi43'):
    r = subprocess.run(['schtasks', '/Query', '/TN', task], capture_output=True, text=True)
    assert r.returncode != 0
    print(task, 'absent', flush=True)
for pid in (4564, 9672):
    r = subprocess.run(['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
                       capture_output=True, text=True)
    assert f'"{pid}"' not in r.stdout, r.stdout
    print('own PID', pid, 'absent', flush=True)
print('CLEANUP_PASS', flush=True)
