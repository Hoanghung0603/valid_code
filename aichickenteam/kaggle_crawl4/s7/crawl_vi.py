"""Kaggle (CPU, internet): crawl CHẬM trang VI còn thiếu (chủ yếu Long Châu). Long Châu chặn IP khi vượt ~500 trang/đợt nhanh
(429→403 ~20h), nhưng chịu được ~450 trang/giờ đều đặn ⇒ --per-domain 1 --delay 8. Kết quả: /kaggle/working/docs_out/corpus-*.parquet."""
import glob
import os
import signal
import subprocess
import sys
import time

W = "/kaggle/working"
SRC = glob.glob("/kaggle/input/**/crawler.tgz", recursive=True)[0]
IDS = glob.glob("/kaggle/input/**/ids_r4_7.csv", recursive=True)[0]
subprocess.run("curl -s https://api.ipify.org; echo", shell=True)  # IP công khai (Long Châu chặn theo IP)
os.makedirs(f"{W}/r2ai", exist_ok=True)
os.chdir(f"{W}/r2ai")
subprocess.run(f"tar xzf {SRC}", shell=True, check=True)
subprocess.run(f"{sys.executable} -m pip -q install curl_cffi==0.16.3 trafilatura==2.2.0 selectolax lxml_html_clean polars zstandard orjson "
               f"charset-normalizer pyyaml huggingface_hub", shell=True)
subprocess.run("hf download AIGuruTinix/ViBioMIR links_corpus.parquet --repo-type dataset --local-dir data/ViBioMIR "
               "|| huggingface-cli download AIGuruTinix/ViBioMIR links_corpus.parquet --repo-type dataset --local-dir data/ViBioMIR",
               shell=True)
subprocess.run("sed -i 's/^  extract_workers: [0-9]*/  extract_workers: 4/; s/^  flush_every: [0-9]*/  flush_every: 5000/' configs/default.yaml",
               shell=True)
p = subprocess.Popen([sys.executable, "-m", "src.crawl", "--ids-file", IDS, "--per-domain", "1", "--delay", "8",
                      "--concurrency", "4", "--out-dir", f"{W}/docs_out", "--state-db", f"{W}/crawl.sqlite"])
t0 = time.time()
while p.poll() is None and time.time() - t0 < 10.5 * 3600:
    time.sleep(60)
if p.poll() is None:
    p.send_signal(signal.SIGINT)          # dừng êm: crawler flush buffer rồi thoát
    try:
        p.wait(timeout=600)
    except subprocess.TimeoutExpired:
        p.kill()
print("files:", len(glob.glob(f"{W}/docs_out/*.parquet")))
subprocess.run(f"rm -rf {W}/r2ai", shell=True)   # chỉ giữ output cần thiết
