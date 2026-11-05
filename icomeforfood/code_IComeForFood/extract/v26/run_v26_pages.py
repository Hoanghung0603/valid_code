"""Run TV's build v26 per-page extraction (notebook 03 at a given commit) on HTML shard records.

Executes the notebook's config/cleaner cells with DATA_ROOT pointed at a scratch folder (hints + domain plan
only, no gold answers), then calls _extract_shard_html_doc on each record, as the shard runner does.

    python -I run_v26_pages.py <notebook.py> <shard.jsonl.zst> <out.jsonl> [postfilter_override.py]
"""
import ast
import base64
import io
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq
import zstandard

sys.stdout.reconfigure(encoding="utf-8")
nb_py, shard, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
override = Path(sys.argv[4]) if len(sys.argv) > 4 else None
DATA = Path(__file__).resolve().parent / "data"

cells = {}
for part in nb_py.read_text(encoding="utf-8").split("# %% CELL ")[1:]:
    num, src = part.split("\n", 1)
    cells[int(num)] = src

g = {"__name__": "__nb__", "display": print, "COLAB": False}
src5 = cells[5].replace("DATA_ROOT = Path('D:/r2ai_data')", f"DATA_ROOT = Path(r'{DATA}')")
assert str(DATA) in src5
for n, src in [(5, src5), (6, cells[6]), (7, cells[7]), (10, cells[10]), (13, cells[13]), (14, cells[14])]:
    exec(compile(src, f"cell{n}", "exec"), g)
if override:  # a candidate postfilter, to test a new rule on the same extractor
    ns = {"__name__": "_override"}
    exec(compile(override.read_text(encoding="utf-8"), str(override), "exec"), ns)
    g["filter_doc"] = ns["filter_doc"]
    print("postfilter override:", override)
# cell 23 holds the shard helpers next to whole-run drivers: take only imports and function defs
tree = ast.parse(cells[23])
keep = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))]
exec(compile(ast.Module(body=keep, type_ignores=[]), "cell23", "exec"), g)
print("build", g.get("GOLD_NOTEBOOK_BUILD"), "postfilter", g["POSTFILTER_SHA1"], "hints", g["CONTAINER_HINTS_SHA1"])

corpus = pq.read_table("D:/r2ai_data/raw/links_corpus_vi_en_zh.parquet").to_pandas().set_index("id")["url"]
raw = zstandard.ZstdDecompressor().stream_reader(open(shard, "rb")).read().decode("utf-8")
recs = [json.loads(l) for l in raw.splitlines() if l.strip()]
with open(out, "w", encoding="utf-8") as f:
    for r in recs:
        body = base64.b64decode(r["body_b64"])
        url = corpus[int(r["id"])]
        doc = g["_extract_shard_html_doc"](body, r.get("encoding") or "utf-8", url, r.get("final_url") or url,
                                           int(r["id"]))
        f.write(json.dumps({k: doc[k] for k in ("id", "url", "title", "text", "quality_flags", "n_chars")}
                           | {"http_code": r.get("http_code"), "n_bytes": len(body),
                              "block_types": [b["type"] for b in doc["blocks"]]}, ensure_ascii=False) + "\n")
print(len(recs), "records ->", out)
