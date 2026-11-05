"""Synthetic dev set: Vietnamese patient questions generated from sampled corpus chunks.

Each query has one source chunk, so qrels have |G| = 1 doc and 1 chunk, while the real gold sets hold ~40-47 docs
per query (inferred from the teammates' 1/2/3-doc probes: P/R = 0.265/0.0065 -> ~41). The set therefore measures
retrieval quality (is the source chunk among the candidates, how high after rerank), not the number of docs
to submit. Chunks are stratified by language (zh ~52% of chunks) and need >= min_tokens so the question has
substance. Query style mirrors the real set: ~55% short title-like questions, ~45% first-person narratives.
Generated with an OpenAI-compatible chat model (Qwen3.6-27B on FPT now: DEV ONLY, never used for submissions).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from medret.retrieve.translate import Translator

STYLE_PROMPT = {
    "short": "one short question (8-20 words), like the title of a question on a Vietnamese health forum",
    "narrative": "a first-person message of 30-70 words from a patient or relative describing their situation "
                 "(age, symptoms, test values, what they already did) and ending with one or two questions",
}
SYSTEM = (
    "You write realistic Vietnamese questions that patients send to doctors. You are given a passage from a "
    "medical web page (Vietnamese or Chinese). Write {style}, in natural Vietnamese with diacritics, that this "
    "passage answers well. Do not copy sentences from the passage, do not mention the passage or the website, "
    "do not translate Chinese terms literally, use the words a Vietnamese patient would use. Output only the question."
)


def make_devset(meta, out_dir: Path, name: str, n: int, model: str, cache_path: Path, seed: int = 0,
                min_tokens: int = 80, zh_share: float = 0.52, narrative_share: float = 0.45,
                id_offset: int = 100_000, threads: int = 8) -> dict:
    con = meta.con
    n_zh = round(n * zh_share)
    rows = []
    for lang, k in (("zh", n_zh), ("vi", n - n_zh)):
        # filter first, then sample: DuckDB applies USING SAMPLE to the FROM clause, i.e. before WHERE
        rows.append(con.execute(f"""
            WITH f AS (SELECT chunk_id, doc_id, lang, start, "end" FROM chunks
                       WHERE lang = '{lang}' AND n_tokens >= {min_tokens}),
                 s AS (SELECT * FROM f USING SAMPLE reservoir({k} ROWS) REPEATABLE ({seed}))
            SELECT s.*, d.title, d.host, d.text FROM s JOIN documents d USING (doc_id)""").df())
    s = pd.concat(rows, ignore_index=True).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    s["chunk_text"] = [t[a:b] for t, a, b in zip(s["text"], s["start"], s["end"], strict=True)]
    s["style"] = ["narrative" if i < round(n * narrative_share) else "short" for i in range(len(s))]
    s["id"] = range(id_offset + 1, id_offset + 1 + len(s))
    s["query"] = ""
    for style, g in s.groupby("style"):
        gen = Translator(model, cache_path=cache_path, threads=threads, system=SYSTEM.format(style=STYLE_PROMPT[style]),
                         prompt_version=f"devq-{style}-v1")
        s.loc[g.index, "query"] = gen.translate(g["chunk_text"].tolist())
    s = s[s["query"].str.len() > 5]
    out_dir.mkdir(parents=True, exist_ok=True)
    s[["id", "query", "style", "lang", "host", "doc_id", "chunk_id"]].to_parquet(
        out_dir / f"{name}_queries.parquet", index=False)
    qrels = [{"id": int(r.id), "relevant_docs": [int(r.doc_id)],
              "relevant_chunks": [{"doc_id": int(r.doc_id), "chunk_text": r.chunk_text}]} for r in s.itertuples()]
    (out_dir / f"{name}.json").write_text(json.dumps(qrels, ensure_ascii=False), encoding="utf-8")
    (out_dir / f"{name}.meta.json").write_text(json.dumps({
        "n": len(s), "model": model, "dev_only": True, "seed": seed, "min_tokens": min_tokens,
        "by_lang": s["lang"].value_counts().to_dict(), "by_style": s["style"].value_counts().to_dict()}, indent=1))
    return {"n": len(s), "by_lang": s["lang"].value_counts().to_dict()}
