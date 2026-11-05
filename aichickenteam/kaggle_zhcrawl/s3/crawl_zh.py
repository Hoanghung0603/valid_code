"""Kaggle CPU (internet): cào bộ 235k doc ZH của file corpus mới (BTC 12/10 – gold ZH nằm trọn trong bộ này; 94/3.8k chunk ZH cũ trúng).
Máy local bị 120ask chặn IP (HTTP 521) ⇒ cào từ IP Kaggle. SHARD = biến môi trường trong tên file ids (zh_ids_<i>.csv) hoặc 'test'.
Out: /kaggle/working/docs_out/corpus-*.parquet + crawl.sqlite."""
import glob, os, signal, subprocess, sys, time
SHARD = "3"
W = "/kaggle/working"; I = os.path.dirname(glob.glob("/kaggle/input/**/crawler.tgz", recursive=True)[0])
subprocess.run("curl -s https://api.ipify.org; echo", shell=True)
os.makedirs(f"{W}/r2ai/data/ViBioMIR", exist_ok=True); os.chdir(f"{W}/r2ai")
subprocess.run(f"tar xzf {I}/crawler.tgz", shell=True, check=True)
subprocess.run(f"cp {I}/links_corpus.parquet data/ViBioMIR/links_corpus.parquet", shell=True, check=True)
subprocess.run(f"{sys.executable} -m pip -q install curl_cffi==0.16.3 trafilatura==2.2.0 selectolax lxml_html_clean polars zstandard orjson charset-normalizer pyyaml", shell=True)
subprocess.run("sed -i 's/^  extract_workers: [0-9]*/  extract_workers: 4/; s/^  flush_every: [0-9]*/  flush_every: 3000/' configs/default.yaml", shell=True)
ids = f"{I}/zh_ids_0.csv" if SHARD == "test" else f"{I}/zh_ids_{SHARD}.csv"
args = [sys.executable, "-m", "src.crawl", "--ids-file", ids, "--priority", "zh-first", "--out-dir", f"{W}/docs_out", "--state-db", f"{W}/crawl.sqlite"]
args += ["--sample-per-domain", "8"] if SHARD == "test" else ["--per-domain", "2", "--delay", "1.0", "--concurrency", "8", "--retry-failed"]
p = subprocess.Popen(args); t0 = time.time()
while p.poll() is None and time.time() - t0 < 11 * 3600:
    time.sleep(60)
if p.poll() is None:
    p.send_signal(signal.SIGINT)
    try: p.wait(timeout=600)
    except subprocess.TimeoutExpired: p.kill()
subprocess.run(f"{sys.executable} - <<'EOF'\nimport sqlite3, pandas as pd\nc = sqlite3.connect('{W}/crawl.sqlite')\nd = pd.read_sql('select host, status from crawl', c)\nprint(d.groupby('host').status.value_counts().unstack(fill_value=0).to_string())\nEOF", shell=True)
print("files:", len(glob.glob(f"{W}/docs_out/*.parquet")))
subprocess.run(f"rm -rf {W}/r2ai", shell=True)
