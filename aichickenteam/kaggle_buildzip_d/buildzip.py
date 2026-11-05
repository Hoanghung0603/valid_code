"""Kaggle CPU (30GB RAM): làm TOÀN BỘ phần nặng tạo bài nộp (máy local OOM/reboot 21–23/10): merge_zh (trộn ZH vào bài VI gốc) → LZMA dict 1GB (song song 2).
Dataset r2ai-buildin: v65.zip.bin (bài VI gốc; đổi đuôi vì Kaggle tự giải nén .zip), ltr_order_oof34(.json/_probs.json), zh_probs*.parquet, zh_docs.parquet,
details_all.jsonl (nhãn), merge_zh.py, ltr.py, rerank_vi.py, zip_lzma.py, jobs.json [{name, zh, scale, zh_chars, cm}]. Out: <name>.zip + sizes.txt."""
import glob, json, os, shutil, subprocess, sys, time
I = os.path.dirname(glob.glob("/kaggle/input/**/jobs.json", recursive=True)[0]); W = "/kaggle/working"; R = "/kaggle/tmp/r"
os.makedirs(f"{R}/scripts", exist_ok=True); os.makedirs(f"{R}/data/eval", exist_ok=True); os.makedirs(f"{R}/data/cache/kg_zhdocs", exist_ok=True); os.makedirs(f"{R}/data/ViBioMIR", exist_ok=True)
for f in ("merge_zh.py", "ltr.py", "rerank_vi.py", "zip_lzma.py"): shutil.copy(f"{I}/{f}", f"{R}/scripts/{f}")
os.symlink(f"{I}/details_all.jsonl", f"{R}/data/eval/details_all.jsonl"); shutil.copy(f"{I}/details_6294_v65.jsonl", f"{R}/data/eval/details_6294_v65.jsonl"); os.symlink(f"{I}/zh_docs.parquet", f"{R}/data/cache/kg_zhdocs/zh_docs.parquet")
shutil.copy(f"{I}/v65.zip.bin", f"{R}/base.zip")
for f in glob.glob(f"{I}/ltr_order_*.json"): shutil.copy(f, f"{R}/{os.path.basename(f)}")
subprocess.run(f"{sys.executable} -m pip -q install pyarrow 2>&1 | tail -1", shell=True)
jobs = json.load(open(f"{I}/jobs.json")); t0 = time.time(); log = lambda *x: print(f"[{time.time()-t0:.0f}s]", *x, flush=True)
order = sorted(glob.glob(f"{R}/ltr_order_*_probs.json"))[0].replace("_probs.json", "")
for j in jobs:
    r = subprocess.run([sys.executable, "scripts/merge_zh.py", "--base", "base.zip", "--order", order, "--zh", f"{I}/{j['zh']}", "--zh-scale", str(j["scale"]),
                        "--zh-chars", str(j["zh_chars"]), "--chunk-max", str(j["cm"])] + (["--vi-chars", str(j["vi_chars"])] if j.get("vi_chars") else []) + ["--out", f"/kaggle/tmp/{j['name']}_d.zip"], cwd=R, capture_output=True, text=True)
    log(j["name"], r.stdout.strip()[-300:], r.stderr.strip()[-300:])
env = dict(os.environ, R2AI_REORDER="1", R2AI_LZMA_DICT_MB="1024", R2AI_LZMA_HC4="1")
pend = [j["name"] for j in jobs if os.path.exists(f"/kaggle/tmp/{j['name']}_d.zip")]
for k in range(0, len(pend), 2):
    ps = [subprocess.Popen([sys.executable, f"{R}/scripts/zip_lzma.py", f"/kaggle/tmp/{n}_d.zip", f"{W}/{n}.zip"], env=env) for n in pend[k:k + 2]]
    [p.wait() for p in ps]; log("nén xong", pend[k:k + 2])
with open(f"{W}/sizes.txt", "w") as o:
    for f in sorted(glob.glob(f"{W}/*.zip")):
        line = f"{os.path.basename(f)} {os.path.getsize(f)}"; print(line); o.write(line + "\n")
