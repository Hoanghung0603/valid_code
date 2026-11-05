"""PubMed Europe PMC records -> docs parquet with build v26's extract_pubmed_doc (spec 5.1), for NT's own use.

    python -I pubmed_docs.py <batch_dir> <out.parquet>
"""
import glob
import json
import os
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

sys.stdout.reconfigure(encoding="utf-8")
src_dir, out = sys.argv[1], Path(sys.argv[2])
here = Path(__file__).resolve().parent
cells = {}
for part in (here / "03_dbee143.py").read_text(encoding="utf-8").split("# %% CELL ")[1:]:
    num, src = part.split("\n", 1)
    cells[int(num)] = src
g = {"__name__": "__nb__", "display": print, "COLAB": False}
src5 = cells[5].replace("DATA_ROOT = Path('D:/r2ai_data')", f"DATA_ROOT = Path(r'{here / 'data'}')")
for n, src in [(5, src5), (6, cells[6]), (7, cells[7]), (10, cells[10]), (13, cells[13]), (14, cells[14])]:
    exec(compile(src, f"cell{n}", "exec"), g)

corpus = pq.read_table("D:/r2ai_data/raw/links_corpus_vi_en_zh.parquet").to_pandas()
pm = corpus[corpus.url.str.contains("pubmed.ncbi.nlm.nih.gov", regex=False)]
id_of = dict(zip(pm.url.str.extract(r"/(\d+)")[0], zip(pm.id, pm.url)))

recs = {}
for f in sorted(glob.glob(os.path.join(src_dir, "*.json.zst"))):
    raw = zstandard.ZstdDecompressor().stream_reader(open(f, "rb")).read()
    for r in (json.loads(raw) if raw.strip() else []):
        recs[str(r.get("pmid") or r.get("id"))] = r
docs = []
for pmid, r in recs.items():
    doc_id, url = id_of[pmid]
    docs.append(g["extract_pubmed_doc"](r, int(doc_id), url))
docs.sort(key=lambda d: d["id"])
bad = [d["id"] for d in docs for b in d["blocks"] if d["text"][b["char_start"]:b["char_end"]] != b["text"]]
out.parent.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pylist(docs), out)
print(f"{len(docs)} docs -> {out}; no_abstract {sum('no_abstract' in d['quality_flags'] for d in docs)}; "
      f"offset errors {len(bad)}; median chars {sorted(d['n_chars'] for d in docs)[len(docs) // 2]}")
