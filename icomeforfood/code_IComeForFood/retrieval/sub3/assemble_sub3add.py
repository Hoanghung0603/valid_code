"""Docs that sub3 adds to sub2: ids not yet in docs_sub2, one copy per id, one parquet per source.

Priority: 27/10 home crawl (25K list, laodong, Long Chau) > TV four-domain git snapshot > 120ask (Kaggle)
> short pages re-extracted from TV Drive HTML. Same doc schema and filters as assemble_sub2.py.
"""
import glob
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
D = Path("D:/r2ai_data")
out = D / "docs_sub3add"
out.mkdir(exist_ok=True)
COLS = ["id", "url", "final_url", "lang", "lang_conf", "title", "published_date", "site", "category", "text", "blocks",
        "n_chars", "sha1_text", "dup_group", "extractor", "quality_flags", "schema_version"]
N = D / "docs_nt/new_1027"
sources = [
    ("home1027", sorted(glob.glob(str(N / "home_*/docs/*.parquet")) + glob.glob(str(N / "ld_a/docs/*.parquet"))
                        + glob.glob(str(N / "lc_a/docs/*.parquet")))),
    ("tv_git1027", sorted(glob.glob(str(N / "tv4_a/docs/*.parquet")))),
    ("120ask", sorted(glob.glob(str(N / "ask*/docs/*.parquet")))),
    ("tv_drive_short", sorted(glob.glob(str(D / "docs_nt/tv_drive_short/*/docs/*.parquet")))),
]
taken: set[int] = set()
for f in glob.glob(str(D / "docs_sub2/*.parquet")):
    taken.update(pq.read_table(f, columns=["id"]).column("id").to_pylist())
print("sub2 ids", len(taken))
total = 0
for name, files in sources:
    n_in = 0
    parts = []
    for f in files:
        t = pq.read_table(f)
        if not t.num_rows:
            continue
        t = t.select([c for c in COLS if c in t.column_names])
        n_in += t.num_rows
        keep = []
        for i, (x, lg, tx) in enumerate(zip(*(t.column(c).to_pylist() for c in ("id", "lang", "text")))):
            if x in taken or lg not in ("vi", "zh", "en") or not (tx or "").strip():
                continue
            taken.add(x)
            keep.append(i)
        parts.append(t.take(pa.array(keep, pa.int64())))
    t = pa.concat_tables(parts, promote_options="default")
    pq.write_table(t, out / f"docs_{name}.parquet")
    total += t.num_rows
    print(f"{name}: {n_in} docs in, {t.num_rows} kept", flush=True)
print("added", total, "-> corpus", len(taken), f"({len(taken) / 1220598:.1%})")
