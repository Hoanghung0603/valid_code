"""Read-only task/lock checks; writes only vi38/bench_out2 verification."""
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('D:/r2ai')
dest = root/'data/sub/vi38/bench_out2'
files = {p.name: dict(bytes=p.stat().st_size,
         sha256=hashlib.sha256(p.read_bytes()).hexdigest())
         for p in sorted(dest.iterdir()) if p.is_file()
         and p.name != 'transfer_verification.json'}
(dest/'transfer_verification.json').write_text(json.dumps(files, indent=2), encoding='utf-8')
task = subprocess.run(['schtasks', '/Query', '/TN', 'r2ai_vi38'], capture_output=True, text=True)
print(json.dumps(dict(files=files, vi38_task_absent=task.returncode != 0,
                     heavy_lock_present=(root/'locks/heavy_job.lock').exists()), indent=2))
