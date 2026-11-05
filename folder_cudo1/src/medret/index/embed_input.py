"""Text fed to BGE-M3 for chunks and queries.

Follows the model's own conventions (checked 2026-10-12):
  - no instruction or prefix on either side: "BGE-M3 no longer requires adding instructions to the queries"
    (huggingface.co/BAAI/bge-m3, FAQ); the official fine-tuning recipe passes no query instruction;
  - passages are "title text" joined by one space: FlagEmbedding's evaluation encodes
    f"{doc['title']} {doc['text']}".strip() (FlagEmbedding/abc/evaluation/searcher.py);
  - dense vector = L2-normalised [CLS] state (paper 2402.03216, sec. 3.2; recipe: --sentence_pooling_method cls
    --normalize_embeddings True); fine-tuning used query_max_len = passage_max_len = 512, pre-training 8192.

We extend "title text" with the chunk's heading path, the same kind of context a title gives, and skip
whichever part the chunk already starts with (the first chunk begins with the <h1>, a section unit with
its own heading). The embedding input is never submitted: chunk_text is always documents.text[start:end].
"""

from __future__ import annotations

import unicodedata

MAX_TITLE_CHARS = 150  # a few junk <title>s are whole paragraphs; keep the prefix a prefix


def _starts_with(text: str, part: str) -> bool:
    return bool(part) and text.lstrip().startswith(part.strip())


def passage_input(title: str, heading_path: str, chunk_text: str) -> str:
    title = " ".join(title.split())[:MAX_TITLE_CHARS]
    parts = []
    if title and not _starts_with(chunk_text, title):
        parts.append(title)
    # heading_path is "H2 > H3"; drop the levels the chunk itself starts with
    heads = [h for h in heading_path.split(" > ") if h] if heading_path else []
    while heads and _starts_with(chunk_text, heads[-1]):
        heads.pop()
    heads = [h for h in heads if h != title]
    if heads:
        parts.append(" > ".join(heads))
    parts.append(chunk_text)
    return " ".join(p.strip() for p in parts if p.strip())


def query_input(query: str) -> str:
    """Queries go in as written (NFC, whitespace collapsed): BGE-M3 takes no query instruction."""
    return " ".join(unicodedata.normalize("NFC", query).split())
