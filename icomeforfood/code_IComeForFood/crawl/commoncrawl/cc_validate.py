"""Are Common Crawl copies of Long Chau pages good enough? Chunk-level check against our home crawl.

Both copies go through build v26 (+ postfilter override) and the s400 chunker. Each home chunk stands in for an
organizer reference chunk; it "passes" when some CC chunk reaches LCS/|ref| >= 0.4 (the official threshold).

    python -I cc_validate.py <pairs.csv> <cc_shard.jsonl.zst> <home_html_dir> <out_dir> [--rebuild-old]
"""
import ast
import base64
import csv
import html
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
import zstandard

sys.stdout.reconfigure(encoding="utf-8")
pairs_csv, cc_shard, home_dir, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
out.mkdir(parents=True, exist_ok=True)
REBUILD = "--rebuild-old" in sys.argv
SP = Path(__file__).resolve().parent
REPO = Path("C:/Users/Admin/OneDrive/Desktop/thi/2026/r2ai")
sys.path[:0] = [str(REPO / "eval"), str(REPO / "chunking")]
from r2ai_eval import overlap, tokenize  # noqa: E402
from chunker import Config, chunk_doc  # noqa: E402

# --- build v26 extractor (same setup as v26/run_v26_pages.py) + postfilter v1.13 from the repo
cells = {}
for part in (SP / "v26" / "03_dbee143.py").read_text(encoding="utf-8").split("# %% CELL ")[1:]:
    num, src = part.split("\n", 1)
    cells[int(num)] = src
DATA = SP / "v26" / "data"
g = {"__name__": "__nb__", "display": print, "COLAB": False}
src5 = cells[5].replace("DATA_ROOT = Path('D:/r2ai_data')", f"DATA_ROOT = Path(r'{DATA}')")
for n, src in [(5, src5), (6, cells[6]), (7, cells[7]), (10, cells[10]), (13, cells[13]), (14, cells[14])]:
    exec(compile(src, f"cell{n}", "exec"), g)
ns = {"__name__": "_override"}
exec(compile((REPO / "extract" / "postfilter.py").read_text(encoding="utf-8"), "postfilter.py", "exec"), ns)
g["filter_doc"] = ns["filter_doc"]
tree = ast.parse(cells[23])
exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.Import, ast.ImportFrom))],
                        type_ignores=[]), "cell23", "exec"), g)
corpus = pq.read_table("D:/r2ai_data/raw/links_corpus_vi_en_zh.parquet").to_pandas().set_index("id")["url"]
cfg = Config.load(REPO / "chunking" / "configs" / "s400.json")


def records(path: Path, want: set[int]) -> dict[int, dict]:
    raw = zstandard.ZstdDecompressor().stream_reader(open(path, "rb")).read().decode("utf-8")
    return {int(r["id"]): r for r in map(json.loads, filter(str.strip, raw.splitlines())) if int(r["id"]) in want}


NEXT = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)


def from_next_data(body: bytes) -> bytes | None:
    """Old Long Chau layout (CC captures before ~2026-04): the article is only in __NEXT_DATA__ JSON."""
    m = NEXT.search(body.decode("utf-8", "replace"))
    d = (json.loads(m.group(1)) if m else {}).get("props", {}).get("pageProps", {}).get("detail", {}).get("data", {})
    if len(d.get("description") or "") < 1000:
        return None
    name, short = html.escape(d.get("name") or ""), html.escape(d.get("shortDescription") or "")
    return (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{name}</title></head><body><main><article>"
            f"<h1>{name}</h1><p>{short}</p>{d['description']}</article></main></body></html>").encode("utf-8")


def extract(r: dict, rebuild_old: bool = False) -> dict:
    i = int(r["id"])
    body = base64.b64decode(r["body_b64"])
    if rebuild_old and r.get("cc_timestamp", "99")[:6] < "202604":
        body = from_next_data(body) or body
    return g["_extract_shard_html_doc"](body, r.get("encoding") or "utf-8", corpus[i],
                                        r.get("final_url") or corpus[i], i)


pairs = list(csv.DictReader(open(pairs_csv, encoding="utf-8")))
ids = {int(p["id"]) for p in pairs}
cc = {}
for f in (sorted(cc_shard.glob("cc_shard_*.jsonl.zst")) if cc_shard.is_dir() else [cc_shard]):
    try:
        cc.update(records(f, ids))
    except Exception as e:  # a shard cut off when the fetch stopped
        print("skip", f.name, type(e).__name__)
home = {}
by_shard = defaultdict(set)
for p in pairs:
    by_shard[p["home_shard_file"]].add(int(p["id"]))
for f, want in by_shard.items():
    home.update(records(home_dir / f, want))
print(f"pairs {len(pairs)}, cc {len(cc)}, home {len(home)}", flush=True)

rows, n_chunks, n_pass, n_pass_doc = [], 0, 0, 0
for p in pairs:
    i = int(p["id"])
    if i not in cc or i not in home:
        continue
    cc[i]["cc_timestamp"] = p["cc_timestamp"]
    dh, dc = extract(home[i]), extract(cc[i], REBUILD)
    ch, cc_ch = chunk_doc(dh, cfg), chunk_doc(dc, cfg)
    cc_tok = [tokenize(c["text"], "ws") for c in cc_ch]
    doc_tok = tokenize(dc["text"], "ws")
    best, best_doc = [], []
    for c in ch:
        ref = tokenize(c["text"], "ws")
        best.append(max((overlap(t, ref) for t in cc_tok), default=0.0))
        best_doc.append(overlap(doc_tok, ref))
    ok = sum(b >= 0.4 for b in best)
    n_chunks += len(ch)
    n_pass += ok
    n_pass_doc += sum(b >= 0.4 for b in best_doc)
    rows.append({"id": i, "url": p["url"], "cc_timestamp": p["cc_timestamp"], "old": p["cc_timestamp"][:6] < "202604", "home_chars": dh["n_chars"],
                 "cc_chars": dc["n_chars"], "home_chunks": len(ch), "cc_chunks": len(cc_ch), "chunks_pass": ok,
                 "pass_rate": round(ok / len(ch), 3) if ch else None,
                 "min_best": round(min(best), 3) if best else None,
                 "mean_best": round(sum(best) / len(best), 3) if best else None})

rows.sort(key=lambda r: (r["pass_rate"] is None, r["pass_rate"] if r["pass_rate"] is not None else 0))
with open(out / ("cc_validate_docs_rebuilt.csv" if REBUILD else "cc_validate_docs.csv"), "w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for flag in (False, True):
    sel = [r for r in rows if r["old"] == flag]
    ch = sum(r["home_chunks"] for r in sel)
    print(f"captures {'< 2026-04' if flag else '>= 2026-04'}: {len(sel)} docs, chunk pass {sum(r['chunks_pass'] for r in sel)}/{ch}")
rates = [r["pass_rate"] for r in rows if r["pass_rate"] is not None]
print(f"docs compared {len(rows)}  (home docs with 0 chunks: {sum(r['home_chunks'] == 0 for r in rows)})")
print(f"home s400 chunks {n_chunks}: matched by a CC s400 chunk >=0.4: {n_pass} ({n_pass / max(n_chunks, 1):.1%}); "
      f"contained in the CC doc text >=0.4: {n_pass_doc} ({n_pass_doc / max(n_chunks, 1):.1%})")
print(f"docs with all chunks passing: {sum(x == 1.0 for x in rates)}/{len(rates)}; "
      f"docs below 0.9: {sum(x < 0.9 for x in rates)}; below 0.5: {sum(x < 0.5 for x in rates)}")
print("worst docs:")
for r in rows[:8]:
    print(" ", r["id"], r["pass_rate"], r["home_chunks"], r["cc_chunks"], r["cc_timestamp"], r["url"][-70:])
