"""Wait read-only for finish task; independently audit only once it exits0."""
import subprocess
import re
import time
from pathlib import Path

base = Path('D:/r2ai/data/sub/vi38')
started = time.monotonic()
while True:
    p = subprocess.run(['schtasks', '/Query', '/TN', 'r2ai_vi38b_finish', '/FO', 'LIST', '/V'],
                       capture_output=True, text=True)
    if not re.search(r'^Status:\s+Running\s*$', p.stdout, re.M):
        break
    assert time.monotonic()-started < 7200
    time.sleep(30)
log = (base/'vi38b_finish.log').read_text(encoding='utf-8', errors='replace')
assert 'CPU DONE finish' in log, log[-5000:]
subprocess.run(['D:/r2ai/.venv/Scripts/python.exe', 'D:/r2ai/scripts/vi38_independent_audit.py'], check=True)
