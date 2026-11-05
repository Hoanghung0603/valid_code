"""Kaggle CPU (30GB RAM): nén LZMA (từ điển 1GB, hc4 ~7.5GB RAM/tiến trình) các zip deflate trong dataset r2ai-zipjob, song song.
Máy local bị OOM/reboot khi nén (21/10). Output /kaggle/working/<tên>_lzma.zip + sizes.txt."""
import glob, os, subprocess, sys, time
I = os.path.dirname(glob.glob("/kaggle/input/**/zip_lzma.py", recursive=True)[0]); W = "/kaggle/working"
env = dict(os.environ, R2AI_REORDER="1", R2AI_LZMA_DICT_MB="1024", R2AI_LZMA_HC4="1")
import zipfile
src = []   # Kaggle tự giải nén .zip trong dataset → <tên>/submission.json; đóng gói lại (STORED) trước khi nén LZMA
for js in sorted(glob.glob(f"{I}/*/submission.json")):
    name = os.path.basename(os.path.dirname(js)); z = f"/kaggle/tmp/{name}.zip"; os.makedirs("/kaggle/tmp", exist_ok=True)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_STORED) as zz: zz.write(js, "submission.json")
    src.append(z)
src += sorted(glob.glob(f"{I}/*.zip"))
print("nén:", [os.path.basename(f) for f in src], flush=True); t0 = time.time()
ps = [(f, subprocess.Popen([sys.executable, f"{I}/zip_lzma.py", f, f"{W}/{os.path.basename(f)[:-4]}_lzma.zip"], env=env)) for f in src[:3]]
for f, p in ps:
    p.wait(); print(os.path.basename(f), "exit", p.returncode, f"{time.time()-t0:.0f}s", flush=True)
with open(f"{W}/sizes.txt", "w") as o:
    for f in sorted(glob.glob(f"{W}/*_lzma.zip")):
        line = f"{os.path.basename(f)} {os.path.getsize(f)}"; print(line); o.write(line + "\n")
