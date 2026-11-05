"""One explicit CPU performance restart; never touches other jobs or stale unknown lock."""
from pathlib import Path
import subprocess

lock=Path('D:/r2ai/locks/heavy_job.lock')
assert lock.read_text()=='vi41 bm25 pid=12076'
subprocess.run(['taskkill','/PID','12076','/T','/F'],check=True)
r=subprocess.run(['tasklist','/FI','PID eq 12076','/NH'],capture_output=True,text=True,check=True)
assert '12076' not in r.stdout,r.stdout
assert lock.read_text()=='vi41 bm25 pid=12076'
lock.unlink()
print('Own BM25 stopped; PID absent; own exact stale lock removed. Resume per-query parts, no retrieval formula changes.')
