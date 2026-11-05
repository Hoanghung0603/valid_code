"""Tải abstract cho mọi URL PubMed trong links_corpus_vi_en_zh.parquet (BTC thêm 240k doc EN, 12/10) vào data/cache/pubmed.sqlite.

Chạy lại được (cache). Sau đó --export ghi data/docs_en/pubmed.parquet (doc_id = id trong links_corpus).
    python scripts/pubmed_fetch_links.py [--export]
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from pubmed import PubMedClient as PubMed  # noqa: E402

LINKS = "data/ViBioMIR_new/links_corpus_vi_en_zh.parquet"


def links():
    n = pd.read_parquet(LINKS)
    n = n[n.url.str.contains("pubmed.ncbi.nlm.nih.gov", regex=False)].copy()
    n["pmid"] = n.url.str.extract(r"/(\d+)/?$")[0]
    return n


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--export", action="store_true"); a = ap.parse_args()
    n = links(); pm = PubMed()
    if a.export:
        docs = {d["doc_id"]: d for d in pm.fetch(n.pmid.tolist(), keep_no_abstract=True)}
        n["title"] = n.pmid.map(lambda p: docs.get(p, {}).get("title", ""))
        n["text"] = n.pmid.map(lambda p: docs.get(p, {}).get("text", ""))
        n["lang"] = "en"
        Path("data/docs_en").mkdir(exist_ok=True)
        n.rename(columns={"id": "doc_id"})[["doc_id", "pmid", "url", "title", "text", "lang"]].to_parquet("data/docs_en/pubmed.parquet")
        print("export", len(n), "with text", (n.text.str.len() > 0).sum(), "with abstract",
              sum(bool(docs.get(p, {}).get("has_abstract")) for p in n.pmid))
        return
    ids, t0 = n.pmid.tolist(), time.time()
    for i in range(0, len(ids), 5000):
        pm.fetch(ids[i:i + 5000], keep_no_abstract=True)
        print(f"{i + 5000}/{len(ids)} {time.time() - t0:.0f}s {pm.stats}", flush=True)


if __name__ == "__main__":
    main()
