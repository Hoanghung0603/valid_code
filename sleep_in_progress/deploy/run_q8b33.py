"""Beam Pod runner cho q8b33: y het run_q8b32.py (da chay dung 26/10), chi doi file text.

Env: CANDS_FILE (ten file trong _up/cands), OUT_FILE, RESUME_DIR (tuong doi Volume /data),
MAX_LEN, BATCH, DTYPE. Text: _up/data/q8b33_docs.jsonl.zst, cau hoi: _up/data/queries.jsonl.
Cands shard chi chua cap CAN cham (co the > 100 cap/cau vi doclist toi 400) -> --top 1000.
"""
import os
import subprocess
import sys
import time

DATA_VOL = "/data"
CODE = "/mnt/code/_up"
HF = "/hf-cache"

for k in ("HF_HUB_CACHE", "HF_HOME", "TRANSFORMERS_CACHE"):
    os.environ[k] = HF

cands = os.environ["CANDS_FILE"]
out_abs = os.path.join(DATA_VOL, os.environ["OUT_FILE"])
resume_abs = os.path.join(DATA_VOL, os.environ["RESUME_DIR"])
os.makedirs(os.path.dirname(out_abs), exist_ok=True)
os.makedirs(resume_abs, exist_ok=True)
scripts = os.path.join(CODE, "scripts")
cmd = [sys.executable, os.path.join(scripts, "qwen_rerank_run.py"),
       "--docs", os.path.join(CODE, "data", "q8b33_docs.jsonl.zst"),
       "--queries", os.path.join(CODE, "data", "queries.jsonl"),
       "--cands", os.path.join(CODE, "cands", cands),
       "--out", out_abs, "--resume-dir", resume_abs,
       "--model", "Qwen/Qwen3-Reranker-8B", "--dtype", os.environ.get("DTYPE", "fp16"),
       "--max-len", os.environ.get("MAX_LEN", "1024"), "--batch", os.environ.get("BATCH", "8"),
       "--top", "1000", "--no-watchdog"]
print("CMD:", " ".join(cmd), flush=True)
t0 = time.time()
r = subprocess.run(cmd, text=True, cwd=scripts)
print(f"exit={r.returncode} elapsed={time.time() - t0:.1f}s", flush=True)
sys.exit(r.returncode)
