"""Helper: read env vars and run qwen_rerank_run.py on Beam Volume."""
import os
import subprocess
import sys
import time

DATA_VOL = "/data"
CODE_VOL = "/mnt/code/_up"
HF_CACHE = "/hf-cache"

cands_file = os.environ.get("CANDS_FILE", "cands_chunkdocs100.json")
out_file = os.environ.get("OUT_FILE", "qrr8b/qwen8b_top100.json")
resume_dir = os.environ.get("RESUME_DIR", "qrr8b/parts")
max_len = int(os.environ.get("MAX_LEN", "1024"))
batch = int(os.environ.get("BATCH", "8"))
dtype = os.environ.get("DTYPE", "fp16")
limit_queries = int(os.environ.get("LIMIT_QUERIES", "0"))

hf = os.path.abspath(HF_CACHE)
os.environ["HF_HUB_CACHE"] = hf
os.environ["HF_HOME"] = hf
os.environ["TRANSFORMERS_CACHE"] = hf

scripts_dir = os.path.join(CODE_VOL, "scripts")
data_dir = os.path.join(CODE_VOL, "data")
out_abs = os.path.join(DATA_VOL, out_file)
resume_abs = os.path.join(DATA_VOL, resume_dir)
os.makedirs(os.path.dirname(out_abs), exist_ok=True)
os.makedirs(resume_abs, exist_ok=True)

cmd = [
    sys.executable,
    os.path.join(scripts_dir, "qwen_rerank_run.py"),
    "--docs",
    os.path.join(data_dir, "chunkdocs100.part000.jsonl.zst"),
    os.path.join(data_dir, "chunkdocs100.part001.jsonl.zst"),
    "--queries", os.path.join(data_dir, "queries.jsonl"),
    "--cands", os.path.join(CODE_VOL, "cands", cands_file),
    "--out", out_abs,
    "--resume-dir", resume_abs,
    "--model", "Qwen/Qwen3-Reranker-8B",
    "--dtype", dtype,
    "--max-len", str(max_len),
    "--batch", str(batch),
    "--top", "100",
    "--no-watchdog",
]
if limit_queries > 0:
    cmd.extend(["--limit-queries", str(limit_queries)])

print("CMD:", " ".join(cmd), flush=True)
print(f"DATA_VOL={DATA_VOL} exists={os.path.isdir(DATA_VOL)}", flush=True)
print(f"scripts_dir={scripts_dir} exists={os.path.isdir(scripts_dir)}", flush=True)
print(f"data_dir={data_dir} exists={os.path.isdir(data_dir)}", flush=True)
t0 = time.time()
r = subprocess.run(cmd, text=True, cwd=scripts_dir)
el = time.time() - t0
print(f"exit={r.returncode} elapsed={el:.1f}s", flush=True)
sys.exit(r.returncode)
