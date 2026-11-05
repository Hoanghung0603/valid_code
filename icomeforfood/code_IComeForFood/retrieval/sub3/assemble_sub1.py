"""Docs for the first (partial-corpus) submission: everything we hold, one copy per id.

Priority when an id appears twice: home crawl > Common Crawl > TV's old shards > zysjonline.
Ids already in the 40K sample (chunked and indexed) and PubMed (our own docs) are left out: those keep their own
chunk/index folders. Writes D:/r2ai_data/docs_sub1/docs_<source>.parquet.
"""
import glob
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
D = Path("D:/r2ai_data")
out = D / "docs_sub1"
out.mkdir(exist_ok=True)
COLS = ["id", "url", "final_url", "lang", "lang_conf", "title", "published_date", "site", "category", "text", "blocks",
        "n_chars", "sha1_text", "dup_group", "extractor", "quality_flags", "schema_version"]
taken = set(pq.read_table(D / "chunks40k/p150/chunks_sample2.parquet", columns=["doc_id"]).column("doc_id").to_pylist())
taken |= set(pq.read_table(D / "docs_nt/pubmed/docs_pubmed.parquet", columns=["id"]).column("id").to_pylist())
print("already covered (40K sample + PubMed):", len(taken))
sources = [("home", sorted(glob.glob(str(D / "docs_nt/home/docs/*.parquet")))),
           ("cc_longchau", sorted(glob.glob(str(D / "crawl_cc/gate/docs/*.parquet")))),
           ("tv_old", sorted(glob.glob(str(D / "from_tv/docs_drive/*.parquet")))),
           ("zysjonline", [str(D / "docs/docs_zysjonline.parquet")])]
total = 0
for name, files in sources:
    tabs = [pq.read_table(f) for f in files]
    tabs = [t.select([c for c in COLS if c in t.column_names]) for t in tabs if t.num_rows]
    t = pa.concat_tables(tabs, promote_options="default")
    ids = t.column("id").to_pylist()
    langs = t.column("lang").to_pylist()
    keep = [i for i, (x, lg) in enumerate(zip(ids, langs)) if x not in taken and lg in ("vi", "zh", "en")]
    t = t.take(keep)
    t = t.take([i for i, x in enumerate(t.column("id").to_pylist())
                if not (x in taken or taken.add(x))])  # first copy of an id within the source wins
    pq.write_table(t, out / f"docs_{name}.parquet")
    total += t.num_rows
    print(f"{name}: {len(ids)} docs in, {t.num_rows} new", flush=True)
print("new docs total", total, "| all ids incl. 40K sample + PubMed:", len(taken))
