"""Build a self-contained Colab notebook that crawls the missing 120ask ids into NT's Google Drive."""
import base64
import json

import duckdb
import zstandard as zstd

D = "D:/r2ai_data/"
REPO = "C:/Users/Admin/OneDrive/Desktop/thi/2026/r2ai/"
rows = duckdb.sql(f"""select m.id, c.url from read_csv('{D}coverage_1026/missing_ids.csv') m
                      join '{D}raw/links_corpus_vi_en_zh.parquet' c using (id)
                      where m.host = 'www.120ask.com' order by m.id""").fetchall()
csv_text = "id,url\n" + "".join(f"{i},{u}\n" for i, u in rows)
open(D + "crawl_120ask/todo_120ask.csv", "w", encoding="utf-8", newline="").write(csv_text)
blob = base64.b64encode(zstd.ZstdCompressor(level=19).compress(csv_text.encode())).decode()
crawler = open(REPO + "crawl/crawl_home.py", encoding="utf-8").read().replace("\r\n", "\n")


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


cells = [
    md(f"""# 120ask crawl (R2AI, NT)

120ask.com refuses Vietnamese IPs but answers from US/EU, so this runs on Colab. It fetches the {len(rows):,} 120ask ids
that nobody has yet, at about 1 request per second (about 17 hours in total), with `crawl/crawl_home.py` from our repo.

Output goes to Google Drive `MyDrive/r2ai_120ask/`: `html/colab_shard_*.jsonl.zst` (raw HTML, same format as the home crawl),
`status.csv` (one row per page) and `crawl.log`. A shard closes every 2,000 pages or 10 minutes, so a disconnect loses at most
10 minutes of work.

**Run:** Runtime → Run all. Cell 3 tests one page first and stops if 120ask refuses Colab.
**Disconnected / 12-hour limit:** open the notebook again and Run all; it continues where it stopped.
**Stop cleanly:** create an empty file `STOP` in `MyDrive/r2ai_120ask/` (or interrupt the last cell)."""),
    code("""from google.colab import drive
drive.mount('/content/drive')
!pip -q install zstandard charset_normalizer brotli httpx
OUT = '/content/drive/MyDrive/r2ai_120ask'
import os; os.makedirs(OUT, exist_ok=True)"""),
    code("""# where Colab is, and whether 120ask answers it
import httpx
print(httpx.get('https://ipinfo.io/json', timeout=20).json())
r = httpx.get('https://www.120ask.com/question/108520700.htm', follow_redirects=True, timeout=30,
              headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36'})
print(r.status_code, r.url, len(r.content), r.headers.get('content-type'))
assert r.status_code == 200 and len(r.content) > 5000, '120ask refuses this Colab IP: tell Claude, do not retry from elsewhere'"""),
    code("%%writefile /content/crawl_home.py\n" + crawler),
    code(f"""# the id,url list (zstd + base64), written to Drive once
import base64, zstandard
todo = OUT + '/todo_120ask.csv'
if not os.path.exists(todo):
    open(todo, 'wb').write(zstandard.ZstdDecompressor().decompress(base64.b64decode(BLOB)))
print(sum(1 for _ in open(todo)) - 1, 'ids to crawl')"""),
    code("""# trial: one page through the real crawler, then look at the row it wrote
!python /content/crawl_home.py --corpus {todo} --out {OUT} --rate 120ask.com=1.0 --workers 1 --limit 1 --shard-prefix colab
!tail -n 2 {OUT}/status.csv"""),
    code("""# the full run: about 1 page/s; rerun this cell (or Run all) after a disconnect
!python /content/crawl_home.py --corpus {todo} --out {OUT} --rate 120ask.com=1.0 --workers 2 --shard-prefix colab --pauses 10 30 60 60"""),
]
cells[4]["source"] = f"BLOB = '{blob}'\n" + cells[4]["source"]
nb = {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
      "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
json.dump(nb, open(D + "crawl_120ask/crawl_120ask_colab.ipynb", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(rows), "ids, notebook", len(json.dumps(nb)) // 1024, "KB")
