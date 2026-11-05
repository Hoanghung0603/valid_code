"""q8b36 tren Modal: Qwen3-Reranker-8B, 4 shard + bench song song, moi shard 1 GPU L40S.

Chay (local):
  modal run --detach deploy/modal_q8b36.py
Input: thu muc staging STAGE (_up/{scripts,data,cands}) giong kho Beam/q8b33.
Output: tra ve truc tiep va ghi OUTDIR/<ten>.json; ban sao + resume tren Volume r2ai-q8b36.
Lenh cham y het run_q8b33.py (Beam): fp16, max-len 1024, batch 8, --top 1000, --no-watchdog.
Toi da 5 GPU (4 shard + bench); ngan chi phi Modal.
"""
import os
from pathlib import Path

import modal

STAGE = Path(os.environ.get("Q8B36_STAGE", __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + r"\diag36\beamproj\_up"))
OUTDIR = Path(os.environ.get("Q8B36_OUT", __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + r"\diag36"))
MODEL = "Qwen/Qwen3-Reranker-8B"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "zstandard==0.23.0", "numpy==1.26.4",
                 "huggingface_hub==0.30.2")
    .env({"HF_HOME": "/hf", "HF_HUB_CACHE": "/hf"})
    .run_commands(f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL}')\"")
    .add_local_dir(str(STAGE), "/root/_up")
)
vol = modal.Volume.from_name("r2ai-q8b36", create_if_missing=True)
app = modal.App("r2ai-q8b36", image=image)


@app.function(gpu="L40S", cpu=4, memory=32768, timeout=4 * 3600, volumes={"/out": vol})
def score(cands_file: str, name: str) -> dict:
    import subprocess
    import sys
    import time
    up = "/root/_up"
    out = f"/out/{name}.json"
    cmd = [sys.executable, f"{up}/scripts/qwen_rerank_run.py",
           "--docs", f"{up}/data/q8b36_docs.jsonl.zst", "--queries", f"{up}/data/queries.jsonl",
           "--cands", f"{up}/cands/{cands_file}", "--out", out, "--resume-dir", f"/out/resume_{name}",
           "--model", MODEL, "--dtype", "fp16", "--max-len", "1024", "--batch", "8",
           "--top", "1000", "--no-watchdog"]
    print("CMD:", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, text=True, cwd=f"{up}/scripts")
    el = round(time.time() - t0, 1)
    vol.commit()
    print(f"exit={r.returncode} elapsed={el}s", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name} exit={r.returncode}")
    return {"name": name, "elapsed": el, "json": Path(out).read_text(encoding="utf-8")}


@app.local_entrypoint()
def main():
    jobs = [(f"cands_q8b36_shard{k}.json", f"q8b36_shard{k}_out") for k in range(4)]
    jobs.append(("cands_q8b36_bench.json", "q8b36_bench_new"))
    tot = 0.0
    for res in score.starmap(jobs):
        p = OUTDIR / f"{res['name']}.json"
        p.write_text(res["json"], encoding="utf-8")
        tot += res["elapsed"]
        print(f"OK {p} {p.stat().st_size} elapsed={res['elapsed']}s", flush=True)
    print(f"GPUSEC={tot:.1f}", flush=True)
