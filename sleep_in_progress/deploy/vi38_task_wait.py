"""Read-only scheduled-task waiter; no relaunch, no lock changes."""
import argparse
import json
from pathlib import Path
import subprocess
import re
import time

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['prepare', 'finish'])
a = ap.parse_args()
base = Path('D:/r2ai/data/sub/vi38')
started = time.monotonic()
while True:
    result = subprocess.run(['schtasks', '/Query', '/TN', 'r2ai_vi38b_'+a.mode, '/FO', 'LIST', '/V'],
                            capture_output=True, text=True)
    if not re.search(r'^Status:\s+Running\s*$', result.stdout, re.M):
        break
    assert time.monotonic()-started < 7200, 'Task wait exceeds2h'
    time.sleep(30)
log = (base/f'vi38b_{a.mode}.log').read_text(encoding='utf-8', errors='replace')
assert f'CPU DONE {a.mode}' in log, log[-5000:]
subprocess.run(['D:/r2ai/.venv/Scripts/python.exe', 'D:/r2ai/scripts/vi38b_status.py'], check=True)
print('TASK DONE', a.mode, flush=True)
