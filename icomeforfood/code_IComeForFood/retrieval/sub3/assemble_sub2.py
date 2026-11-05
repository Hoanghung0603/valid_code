"""Docs for submission 2 (near-full corpus), one copy per id, written as one parquet per source.

Priority when an id appears twice: home crawl > TV git snapshot > TV Drive HTML (all v26 + postfilter 1.13)
> Common Crawl Long Chau > PubMed (TV's 240K minus empty placeholders, else ours) > zysjonline > TV old shards.
"""
import glob
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
D = Path("D:/r2ai_data")
out = D / "docs_sub2"
out.mkdir(exist_ok=True)
COLS = ["id", "url", "final_url", "lang", "lang_conf", "title", "published_date", "site", "category", "text", "blocks",
        "n_chars", "sha1_text", "dup_group", "extractor", "quality_flags", "schema_version"]
sources = [
    ("home", sorted(glob.glob(str(D / "docs_nt/home/docs/*.parquet")))),
    ("tv_git", sorted(glob.glob(str(D / "docs_nt/tv_git_1026/docs/*.parquet")))),
    ("tv_drive", sorted(glob.glob(str(D / "docs_nt/tv_drive_html/w*/docs/*.parquet")))),
    ("cc_longchau", sorted(glob.glob(str(D / "crawl_cc/gate/docs/*.parquet")))),
    ("pubmed_tv", sorted(glob.glob(str(D / "from_tv/audit_1026/docs_pubmed_20261023/*.parquet")))),
    ("pubmed_nt", [str(D / "docs_nt/pubmed/docs_pubmed.parquet")]),
    ("zysjonline", [str(D / "docs/docs_zysjonline.parquet")]),
    ("tv_old", [str(D / "docs_sub1/docs_tv_old.parquet")]),
]
taken: set[int] = set()
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
        ids, langs = t.column("id").to_pylist(), t.column("lang").to_pylist()
        texts = t.column("text").to_pylist()
        keep = []
        for i, (x, lg, tx) in enumerate(zip(ids, langs, texts)):
            if x in taken or lg not in ("vi", "zh", "en") or not (tx or "").strip():
                continue
            taken.add(x)
            keep.append(i)
        parts.append(t.take(pa.array(keep, pa.int64())))
    t = pa.concat_tables(parts, promote_options="default")
    pq.write_table(t, out / f"docs_{name}.parquet")
    total += t.num_rows
    print(f"{name}: {n_in} docs in, {t.num_rows} kept", flush=True)
print("docs total", total)
