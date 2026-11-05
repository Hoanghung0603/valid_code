"""Build v26 docs for Common Crawl Long Chau shards and a per-page quality gate.

    python -I cc_gate.py <html_dir> <cc_usable.csv | crawl_home status.csv> <out_dir> [shard_glob]

Writes <out_dir>/docs/docs_cc_<shard>.parquet (v26 doc schema) and <out_dir>/gate.csv, one row per page:
id, list, cc_timestamp, cc_rebuilt, usable, n_chars, n_blocks, title, flags, gate (ok / reason).
Resumable: shards whose parquet exists are skipped.
"""
import ast
import base64
import csv
import io
import json
import os
import re
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

sys.stdout.reconfigure(encoding="utf-8")
src, usable_csv, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
pattern = sys.argv[4] if len(sys.argv) > 4 else "cc_shard_*.jsonl.zst"
(out / "docs").mkdir(parents=True, exist_ok=True)
SP = Path(__file__).resolve().parent
REPO = Path(os.environ.get("R2AI_REPO", "C:/Users/Admin/OneDrive/Desktop/thi/2026/r2ai"))

cells = {}
for part in (SP / "v26" / "03_dbee143.py").read_text(encoding="utf-8").split("# %% CELL ")[1:]:
    num, s = part.split("\n", 1)
    cells[int(num)] = s
g = {"__name__": "__nb__", "display": lambda *a, **k: None, "COLAB": False}
src5 = cells[5].replace("DATA_ROOT = Path('D:/r2ai_data')", f"DATA_ROOT = Path(r'{SP / 'v26' / 'data'}')")
for n, s in [(5, src5), (6, cells[6]), (7, cells[7]), (10, cells[10]), (13, cells[13]), (14, cells[14])]:
    exec(compile(s, f"cell{n}", "exec"), g)
ns = {"__name__": "_override"}
exec(compile((REPO / "extract" / "postfilter.py").read_text(encoding="utf-8"), "postfilter.py", "exec"), ns)
g["filter_doc"] = ns["filter_doc"]
tree = ast.parse(cells[23])
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))],
                        type_ignores=[]), "cell23", "exec"), g)
corpus = pq.read_table(os.environ.get("R2AI_CORPUS", "D:/r2ai_data/raw/links_corpus_vi_en_zh.parquet")).to_pandas().set_index("id")["url"]
meta = {} if str(usable_csv) == "-" else {int(r["id"]): r for r in csv.DictReader(open(usable_csv, encoding="utf-8"))}
# a crawl_home status.csv instead of cc_usable.csv: keep only the ids whose last row is ok
allowed = {i for i, r in meta.items() if r.get("status") == "ok"} if meta and "status" in next(iter(meta.values())) else None

ERROR_PAGE = re.compile(r"không tìm thấy trang|trang không tồn tại|404 not found|page not found|đã xảy ra lỗi", re.I)
MIN_CHARS = 400
KEEP = set(os.environ.get("R2AI_KEEP", "ok").split(","))  # verdicts whose docs are written


def gate(doc: dict) -> str:
    text = doc.get("text") or ""
    if not text.strip():
        return "empty"
    if doc["n_chars"] < MIN_CHARS:
        return "short"
    if doc["n_chars"] < 2000 and ERROR_PAGE.search(text):
        return "error_page"
    return "ok"


gate_f = out / "gate.csv"
new = not gate_f.exists()
with open(gate_f, "a", encoding="utf-8", newline="") as gf:
    w = csv.writer(gf)
    if new:
        w.writerow(["id", "list", "cc_timestamp", "cc_rebuilt", "usable", "n_chars", "n_blocks", "title", "flags", "gate"])
    for shard in sorted(src.glob(pattern)):
        dst = out / "docs" / f"docs_{shard.name.split('.')[0]}.parquet"
        if dst.exists():
            continue
        # stream the shard line by line: a full shard of raw HTML can be several GB once decompressed
        stream = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(shard, "rb")), encoding="utf-8")
        docs, rows = [], []
        for line in filter(str.strip, stream):
            r = json.loads(line)
            i = int(r["id"])
            if allowed is not None and i not in allowed:
                continue
            if not meta and int(r.get("http_code") or 200) != 200:  # no status file: trust the record's own code
                continue
            m = meta.get(i, {})
            try:
                d = g["_extract_shard_html_doc"](base64.b64decode(r["body_b64"]), r.get("encoding") or "utf-8",
                                                 corpus[i], r.get("final_url") or corpus[i], i)
                verdict = gate(d)
            except Exception as e:  # noqa: BLE001
                d, verdict = {"n_chars": 0, "blocks": [], "title": "", "quality_flags": []}, f"extract_error:{type(e).__name__}"
            if verdict in KEEP:
                docs.append(d)
            rows.append([i, m.get("list"), m.get("cc_timestamp") or r.get("cc_timestamp"), r.get("cc_rebuilt"),
                         m.get("usable"), d.get("n_chars", 0), len(d.get("blocks") or []), (d.get("title") or "")[:120],
                         "|".join(d.get("quality_flags") or []), verdict])
        pq.write_table(pa.Table.from_pylist(docs), dst)
        w.writerows(rows)
        gf.flush()
        print(f"{shard.name}: {len(rows)} pages, {len(docs)} ok, "
              f"rejected {sum(r[-1] != 'ok' for r in rows)}", flush=True)
print("ALL_DONE", flush=True)
