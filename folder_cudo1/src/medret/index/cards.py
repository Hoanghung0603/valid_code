"""Topic cards: a short doc-level text of what a page is about, and a BM25 index over them.

card = title | description | keywords | section | breadcrumb  (parse/metadata.py, site-wide boilerplate removed)
       + URL slug words (unaccented, e.g. "viem tuyen le").
Indexed terms = vi_terms of the card + vi_terms of its accent-stripped copy (so the slug and accent-less queries
match); one Tantivy entry per doc, stored under the doc's head chunk id (doc_id * CHUNK_ID_STRIDE) so the usual
chunk_id -> doc_id mapping holds. Queries are expanded the same way (`card_terms`).
"""

from __future__ import annotations

import json
import time
import unicodedata
from pathlib import Path

import pandas as pd

from medret.chunk.run import CHUNK_ID_STRIDE
from medret.index.lexical import META_FILE, _open, tokenizer_version, vi_terms

CARD_FILE = "cards_vi.parquet"


def strip_accents(text: str) -> str:
    s = unicodedata.normalize("NFD", text.replace("đ", "d").replace("Đ", "D"))
    return "".join(ch for ch in s if unicodedata.category(ch) != "Mn")


def card_terms(text: str) -> list[str]:
    terms = vi_terms(text)
    plain = strip_accents(text)
    return terms + [t for t in vi_terms(plain) if t not in set(terms)]


def build_cards(documents: pd.DataFrame, metadata: pd.DataFrame, out_file: Path) -> pd.DataFrame:
    """documents: doc_id, title; metadata: doc_id, description, keywords, section, breadcrumb, slug."""
    d = documents[["doc_id", "title"]].merge(metadata, how="left", on="doc_id").fillna("")
    parts = ["title", "description", "keywords", "section", "breadcrumb"]
    d["card"] = d[parts].apply(lambda r: " | ".join(x for x in r if x), axis=1)
    d["card"] = d["card"].str.cat(d["slug"], sep=" | ")
    cards = d[["doc_id", "card"]]
    cards.to_parquet(out_file, index=False)
    return cards


def build_card_bm25(cards: pd.DataFrame, out_dir: Path, log=print) -> int:
    import shutil

    import tantivy

    if out_dir.exists():
        shutil.rmtree(out_dir)
    index = _open(out_dir, create=True)
    writer = index.writer(heap_size=300_000_000, num_threads=2)  # laptop RAM budget
    t0 = time.time()
    for i, (doc_id, card) in enumerate(zip(cards["doc_id"], cards["card"], strict=True), 1):
        writer.add_document(tantivy.Document(chunk_id=int(doc_id) * CHUNK_ID_STRIDE, body=" ".join(card_terms(card))))
        if i % 200_000 == 0:
            log(f"[card bm25] {i:,}/{len(cards):,} in {time.time() - t0:.0f}s")
    writer.commit()
    writer.wait_merging_threads()
    (out_dir / META_FILE).write_text(json.dumps({"lang": "vi", "n_docs": len(cards), "tokenizer": tokenizer_version("vi"),
                                                 "kind": "topic-card"}))
    log(f"[card bm25] {len(cards):,} cards indexed in {time.time() - t0:.0f}s -> {out_dir}")
    return len(cards)
