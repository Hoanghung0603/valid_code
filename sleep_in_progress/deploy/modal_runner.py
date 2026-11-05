r"""GPU Modal cho runner private (theo khan deploy/modal_q8b33.py): rr | docwin | q8b | bench_bge.

Chay (local, venv <HIST_TMP>\modalenv):
  modal run --detach deploy/modal_runner.py --stage rr --tag rsmoke20
  modal run --detach deploy/modal_runner.py --stage docwin --tag rsmoke20
  modal run --detach deploy/modal_runner.py --stage q8b --tag rsmoke20 --cands cands_q8b_shard0.json --out q8b_shard0_out.json
  modal run --detach deploy/modal_runner.py --stage bench_bge --tag rsmoke20
Bench 8B: chay stage q8b voi --cands cands_q8b_bench.json --out q8b_bench_out.json (cung lenh cham).
Input:  Volume r2ai-runner /in/<tag>/ (queries.jsonl, docs_*.jsonl.zst, cands*.json, bench_*_in.json)
Output: Volume r2ai-runner /out/<tag>/ (rerank_pool*.json, docwin.json.zst, q8b_*_out.json,
        bench_*_out.json, job_report_*.json). Lenh cham y nhu pipeline: rerank/docwin bge fp16
max-len 512 batch 16; 8B fp16 max-len 1024 batch 8 --top 1000 --no-watchdog (modal_q8b33).
"""
import os
from pathlib import Path

import modal

MODEL_BGE = "BAAI/bge-reranker-v2-m3"
MODEL_8B = "Qwen/Qwen3-Reranker-8B"
SCRIPTS = Path(os.environ.get("RUNNER_SCRIPTS", __import__("os").environ.get("WORKDIR", "<WORKDIR>") + r"\r2ai\scripts"))

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "transformers==4.51.3", "zstandard==0.23.0", "numpy==1.26.4",
                 "huggingface_hub==0.30.2")
    .env({"HF_HOME": "/hf", "HF_HUB_CACHE": "/hf"})
    .run_commands(f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL_8B}')\"")
    .run_commands(f"python -c \"from huggingface_hub import snapshot_download; snapshot_download('{MODEL_BGE}')\"")
    .add_local_dir(str(SCRIPTS), "/root/scripts")
)
vol = modal.Volume.from_name("r2ai-runner", create_if_missing=True)
app = modal.App("r2ai-runner", image=image)


@app.function(gpu="L40S", cpu=4, memory=32768, timeout=4 * 3600, volumes={"/vol": vol})
def rr(tag: str, cands: str = "pool_cands.json", out: str = "rerank_pool.json") -> dict:
    import subprocess
    import sys
    import time
    from pathlib import Path
    docs = sorted(str(p) for p in Path(f"/vol/in/{tag}").glob("docs_rr.part*.jsonl.zst"))
    cmd = [sys.executable, "/root/scripts/rerank_run.py", "--docs", *docs,
           "--queries", f"/vol/in/{tag}/queries.jsonl", "--cands", f"/vol/in/{tag}/{cands}",
           "--out", f"/vol/out/{tag}/{out}", "--max-len", "512", "--batch", "16",
           "--top", "800", "--resume-dir", f"/vol/out/{tag}/resume_{out.replace('.', '_')}"]
    print("CMD:", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, text=True)
    el = round(time.time() - t0, 1)
    if r.returncode != 0:
        raise RuntimeError(f"rr exit={r.returncode}")
    rep = {"name": "rr", "elapsed": el, "gpus": 1, "cands": cands, "out": out}
    import json
    Path(f"/vol/out/{tag}/job_report_{Path(out).stem}.json").write_text(json.dumps(rep), encoding="utf-8")
    vol.commit()
    return rep


@app.function(gpu="L40S", cpu=4, memory=32768, timeout=4 * 3600, volumes={"/vol": vol})
def docwin(tag: str, cands: str = "cands_docwin.json", out: str = "docwin.json.zst") -> dict:
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path
    docs = sorted(str(p) for p in Path(f"/vol/in/{tag}").glob("docs_docwin.part*.jsonl.zst"))
    cmd = [sys.executable, "/root/scripts/docwin_run.py", "--docs", *docs,
           "--queries", f"/vol/in/{tag}/queries.jsonl", "--cands", f"/vol/in/{tag}/{cands}",
           "--out", f"/vol/out/{tag}/{out}", "--max-len", "512", "--batch", "16",
           "--top-docs", "500", "--window-words", "300", "--window-stride", "250",
           "--resume-dir", f"/vol/out/{tag}/resume_{out.replace('.', '_')}"]
    print("CMD:", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, text=True)
    el = round(time.time() - t0, 1)
    if r.returncode != 0:
        raise RuntimeError(f"docwin exit={r.returncode}")
    rep = {"name": "docwin", "elapsed": el, "gpus": 1, "cands": cands, "out": out}
    Path(f"/vol/out/{tag}/job_report_{Path(out).stem}.json").write_text(json.dumps(rep), encoding="utf-8")
    vol.commit()
    return rep


@app.function(gpu="L40S", cpu=4, memory=32768, timeout=4 * 3600, volumes={"/vol": vol})
def q8b(tag: str, cands: str = "cands_q8b_shard0.json", out: str = "q8b_shard0_out.json") -> dict:
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path
    prefix = {"cands_q8b_r2.json": "docs_q8b_r2", "cands_q8b_r3.json": "docs_q8b_r3"}.get(cands, "docs_q8b")
    docs = sorted(str(p) for p in Path(f"/vol/in/{tag}").glob(f"{prefix}.part*.jsonl.zst"))
    assert docs, (tag, prefix)
    cmd = [sys.executable, "/root/scripts/qwen_rerank_run.py", "--docs", *docs,
           "--queries", f"/vol/in/{tag}/queries.jsonl", "--cands", f"/vol/in/{tag}/{cands}",
           "--out", f"/vol/out/{tag}/{out}", "--resume-dir", f"/vol/out/{tag}/resume_{out.replace('.', '_')}",
           "--model", MODEL_8B, "--dtype", "fp16", "--max-len", "1024", "--batch", "8",
           "--top", "1000", "--no-watchdog"]
    print("CMD:", " ".join(cmd), flush=True)
    t0 = time.time()
    r = subprocess.run(cmd, text=True)
    el = round(time.time() - t0, 1)
    if r.returncode != 0:
        raise RuntimeError(f"q8b exit={r.returncode}")
    rep = {"name": "q8b", "elapsed": el, "gpus": 1, "cands": cands, "out": out}
    Path(f"/vol/out/{tag}/job_report_{Path(out).stem}.json").write_text(json.dumps(rep), encoding="utf-8")
    vol.commit()
    return rep


@app.function(gpu="L40S", cpu=4, memory=32768, timeout=2 * 3600, volumes={"/vol": vol})
def bench_bge(tag: str) -> dict:
    """Cham lai mau cap public (bench_rr_in.json, bench_docwin_in.json) bang bge-reranker-v2-m3."""
    import json
    import sys
    import time
    from pathlib import Path
    sys.path.insert(0, "/root/scripts")
    from rerank_run import load_model, score_pairs
    Path(f"/vol/out/{tag}").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok, model = load_model(MODEL_BGE)
    rep = {"name": "bench_bge", "gpus": 1}
    for fn in ("bench_rr_in", "bench_docwin_in"):
        p = Path(f"/vol/in/{tag}/{fn}.json")
        if not p.exists():
            continue
        rows = json.loads(p.read_text(encoding="utf-8"))
        scores = score_pairs(model, tok, [(r["query"], r["text"]) for r in rows], 512, 16)
        Path(f"/vol/out/{tag}/{fn.replace('_in', '_out')}.json").write_text(
            json.dumps({"scores": [round(float(s), 4) for s in scores]}), encoding="utf-8")
        rep[fn] = len(rows)
    rep["elapsed"] = round(time.time() - t0, 1)
    Path(f"/vol/out/{tag}/job_report_bench_bge.json").write_text(json.dumps(rep), encoding="utf-8")
    vol.commit()
    return rep


STAGE_FN = {"rr": rr, "docwin": docwin, "q8b": q8b, "bench_bge": bench_bge}


@app.local_entrypoint()
def main(stage: str = "rr", tag: str = "run", cands: str = "", out: str = ""):
    if stage not in STAGE_FN:
        raise SystemExit(f"stage la: {sorted(STAGE_FN)}")
    fn = STAGE_FN[stage]
    kw = {}
    if stage in ("rr", "docwin", "q8b"):
        kw = {"cands": cands or ("pool_cands.json" if stage == "rr"
                                 else "cands_docwin.json" if stage == "docwin"
                                 else "cands_q8b_shard0.json"),
              "out": out or ("rerank_pool.json" if stage == "rr"
                             else "docwin.json.zst" if stage == "docwin" else "q8b_shard0_out.json")}
    res = fn.remote(tag, **kw)
    print("OK", res, flush=True)
