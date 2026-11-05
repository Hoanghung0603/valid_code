"""Import the Vietnamese docs that only the team snapshot has, as parsed documents for our pipeline.

  python scripts/analysis/import_team_snapshot.py   # -> data/parsed/documents/<site>/part-gloryvu-20261019.parquet
  python scripts/analysis/import_team_snapshot.py --src gloryvu_20261023 --lang zh
      # Chinese docs we have no text for (2026-10-26: the public gold has zh docs) -> part-gloryvu-20261023-zh.parquet
Source: Kaggle private dataset gloryvu/vibiomir-crawl-snapshot (taken 2026-10-19 12:25), downloaded to
data/external/gloryvu_20261019/; doc ids in new_vi_doc_ids.parquet (41,240 corpus-v2 Vietnamese docs that our crawl has
no text for). Per the snapshot README, the chunks of a doc in chunk_idx order are its whole page text (`context` is
verbatim page text), so doc text = contexts joined by "\\n" and each chunk becomes one block (kind "p"; its section
title as heading_path when it differs from the page title). Our chunker re-splits the blocks to its own size, and
submitted chunk_text stays a slice of this text. Extraction differs from our parser: `fallback_cov` docs (~6k of the
41k) keep sidebar teasers. Stand-in rows of extra_chunks.parquet (URL / title only) are not imported.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from medret.config import get_config
from medret.parse.document import BLOCK_SEP
from medret.parse.run import DOC_SCHEMA, DOCS_SUBDIR

SRC = "gloryvu_20261019"
VERSION = "gloryvu-snapshot-20261019"
PART = "part-gloryvu-20261019.parquet"


def build_docs(chunks: pd.DataFrame, lang: str = "vi", version: str = VERSION) -> list[dict]:
    out = []
    for doc_id, g in chunks.sort_values(["doc_id", "chunk_idx"]).groupby("doc_id", sort=False):
        texts = [str(t) for t in g["context"]]
        first = g.iloc[0]
        cands = (first["page_title"], first["h1"], first["title"])
        title = next((str(x) for x in cands if isinstance(x, str) and x.strip()), "")
        blocks, pos = [], 0
        for i, (t, sec) in enumerate(zip(texts, g["title"], strict=True)):
            if i:
                pos += len(BLOCK_SEP)
            head = str(sec) if isinstance(sec, str) and sec.strip() and sec.strip() != title.strip() else ""
            blocks.append({"start": pos, "end": pos + len(t), "kind": "p", "heading_path": head})
            pos += len(t)
        text = BLOCK_SEP.join(texts)
        assert len(text) == pos
        out.append({"doc_id": int(doc_id), "url": str(first["url"]), "host": str(first["site"]), "lang": lang,
                    "title": title, "text": text, "blocks": blocks, "n_chars": len(text),
                    "text_sha1": hashlib.sha1(text.encode("utf-8")).hexdigest(), "dup_group": None,
                    "extractor_version": f"{version}:{first['method']}"})
    return out


def zh_missing(src: Path) -> set[int]:
    """Snapshot zh docs that the meta store has no text for."""
    from medret.index.meta import DB_FILE, MetaStore

    ids = set(pd.concat(pd.read_parquet(f, columns=["doc_id"]) for f in glob.glob(str(src / "chunks_zh-*.parquet")))
              ["doc_id"].tolist())
    have = set(MetaStore(get_config().paths.index_dir / DB_FILE).documents(ids, with_text=False)["doc_id"].tolist())
    return ids - have


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--lang", default="vi", choices=["vi", "zh"])
    a = ap.parse_args()
    cfg = get_config()
    src = cfg.paths.data_dir / "external" / a.src
    version, part = (VERSION, PART) if a.lang == "vi" else (f"gloryvu-snapshot-{a.src[-8:]}",
                                                            f"part-gloryvu-{a.src[-8:]}-zh.parquet")
    want = (set(pd.read_parquet(src / "new_vi_doc_ids.parquet")["doc_id"].tolist()) if a.lang == "vi"
            else zh_missing(src))
    cols = ["doc_id", "url", "chunk_idx", "title", "context", "method", "page_title", "h1", "site"]
    docs_dir = cfg.paths.parsed_dir / DOCS_SUBDIR
    # only the wanted rows are materialised (~6% of the snapshot), and all files are gathered before assembly, so a
    # doc whose chunks straddle row groups or files is still assembled in full
    parts = [pq.read_table(f, columns=cols, filters=[("doc_id", "in", sorted(want))]).to_pandas()
             for f in sorted(glob.glob(str(src / f"chunks_{a.lang}-*.parquet")))]
    chunks = pd.concat(parts, ignore_index=True)
    del parts
    assert not chunks.duplicated(["doc_id", "chunk_idx"]).any(), "duplicate (doc_id, chunk_idx) rows in the snapshot"
    by_site: dict[str, list[dict]] = {}
    for d in build_docs(chunks, a.lang, version):
        by_site.setdefault(d["host"], []).append(d)
    n = sum(len(ds) for ds in by_site.values())
    for site, ds in by_site.items():
        out = docs_dir / site / part
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        pq.write_table(pa.Table.from_pylist(sorted(ds, key=lambda d: d["doc_id"]), schema=DOC_SCHEMA), tmp,
                       compression="zstd")
        Path(tmp).replace(out)
    print(f"imported {n:,} docs into {len(by_site)} sites (missing from the snapshot: {len(want) - n:,}) "
          f"-> {docs_dir}/*/{part}")


if __name__ == "__main__":
    main()
