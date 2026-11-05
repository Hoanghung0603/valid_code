"""Retrieval diagnostics of a run on a synthetic dev set (one source doc / chunk per query).

  candidates  recall of the source doc / chunk within the RRF top-k, overall and per channel and source language
  rerank      rank of the source doc after reranking (doc score = best chunk score): recall@k and MRR
This is the right lens for |G| = 1 qrels; the official F2 on such qrels mostly reflects how many docs are submitted.
"""

from __future__ import annotations

import pandas as pd

KS = (1, 5, 10, 20, 50, 100, 300)


def _first_rank(sorted_ids: pd.Series, target) -> int | None:
    hit = sorted_ids.index[sorted_ids == target]
    return int(hit[0]) if len(hit) else None


def retrieval_report(run_dir, dev_queries: pd.DataFrame) -> dict:
    cands = pd.read_parquet(run_dir / "candidates.parquet")
    reranked = pd.read_parquet(run_dir / "reranked.parquet")
    src = dev_queries.set_index("id")
    rows = []
    for qid, g in cands.groupby("query_id"):
        if qid not in src.index:
            continue
        s = src.loc[qid]
        g = g.sort_values("rrf", ascending=False).reset_index(drop=True)
        chunk_rank = _first_rank(g["chunk_id"], s["chunk_id"])
        doc_ids = g["chunk_id"] // 10_000  # chunk_id = doc_id * 10_000 + index
        doc_rank_cand = _first_rank(doc_ids.drop_duplicates().reset_index(drop=True), s["doc_id"])
        routes = g.loc[doc_ids == s["doc_id"], "routes"].str.cat(sep="|")
        r = reranked[reranked["query_id"] == qid]
        docs = r.groupby("doc_id")["score"].max().sort_values(ascending=False).reset_index()
        doc_rank_rr = _first_rank(docs["doc_id"], s["doc_id"])
        rows.append({"id": qid, "lang": s["lang"], "style": s["style"], "chunk_rank": chunk_rank,
                     "doc_rank_cand": doc_rank_cand, "doc_rank_rerank": doc_rank_rr,
                     **{f"via_{ch}": ch in routes for ch in ("bm25_vi", "bm25_zh", "dense_vi", "dense_zh")}})
    df = pd.DataFrame(rows)

    def recall(col: str, k: int, frame: pd.DataFrame) -> float:
        return float((frame[col].notna() & (frame[col] < k)).mean()) if len(frame) else 0.0

    def block(frame: pd.DataFrame) -> dict:
        return {
            "n": len(frame),
            "doc_recall_cand": {k: round(recall("doc_rank_cand", k, frame), 3) for k in KS},
            "chunk_recall_cand@300": round(recall("chunk_rank", 300, frame), 3),
            "doc_recall_rerank": {k: round(recall("doc_rank_rerank", k, frame), 3) for k in KS[:5]},
            "mrr_rerank": round(float(frame["doc_rank_rerank"].map(lambda x: 0 if pd.isna(x) else 1 / (x + 1))
                                      .mean()), 3) if len(frame) else 0.0,
            "found_by": {c[4:]: round(float(frame[c].mean()), 3) for c in frame.columns if c.startswith("via_")},
        }

    return {"all": block(df), "by_lang": {k: block(g) for k, g in df.groupby("lang")},
            "by_style": {k: block(g) for k, g in df.groupby("style")}}
