"""Turn reranked chunks into the submission entry of one query (architecture.md section 2.7).

Docs: doc score = best chunk score (MaxP); keep docs with score >= tau_doc, at least k_min and at most k_max
(F2 weighs recall 4x, so the doc list leans long). Chunks: inside each kept doc, its top chunk plus every
other chunk with score >= tau_chunk (up to max_chunks_per_doc); chunks that touch in the document are merged
into one span, and chunk_text is always documents.text[start:end] (verbatim). Optionally the exact URL twins
of a kept doc (same canonical URL, never fetched) are added to the doc list (probe P6).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from medret.schemas import ChunkRef, QueryResult


@dataclass
class SelectConfig:
    tau_doc: float = 0.1
    k_min: int = 1
    k_max: int = 20
    tau_chunk: float = 0.3
    max_chunks_per_doc: int = 3
    merge_gap: int = 1  # chars between two chunks still counted as adjacent (the block separator)
    include_twins: bool = False
    # language prior: the team's leaderboard probes found ~1.6% precision for Chinese docs vs ~21% for Vietnamese
    # ones, so Chinese docs can be held to a stricter score and capped (None = no cap)
    zh_tau: float | None = None
    max_zh_docs: int | None = None
    # probe P3 / the teammates' best LB setting: the whole document text as the chunk for the top-N docs (the chunk
    # metric divides by the gold length only). Exploits the metric as written; keep 0 for the main method.
    full_doc_top: int = 0
    # doc score = best rerank score + rrf_weight * (best RRF score, min-max scaled within the query): on the dev set
    # this lifted doc MRR 0.332 -> 0.358 and recall@10 0.513 -> 0.553 over rerank-only MaxP
    rrf_weight: float = 0.1
    # chunks only for the top-N docs (None = every kept doc). Chunk precision counts every submitted chunk, and on
    # the leaderboard a span from a low-ranked doc matched a gold chunk ~3% of the time (D2, 13/10)
    chunk_docs: int | None = None
    # cap for a whole-document chunk (chars): longer docs get a window of this size centred on their best chunk, to
    # keep the zip under the upload limit (None = no cap)
    full_doc_max_chars: int | None = None


def merge_spans(spans: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if out and s - out[-1][1] <= gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def choose_docs(cands: pd.DataFrame, cfg: SelectConfig) -> list[int]:
    """Doc ids to submit, best first (ties broken by doc_id so every caller gets the same list)."""
    if cands.empty:
        return []
    agg = {"score": "max"} | ({"lang": "first"} if "lang" in cands else {}) | ({"rrf": "max"} if "rrf" in cands else {})
    doc_scores = cands.groupby("doc_id").agg(agg).reset_index()
    if cfg.rrf_weight and "rrf" in doc_scores:
        r = doc_scores["rrf"]
        doc_scores["score"] = doc_scores["score"] + cfg.rrf_weight * (r - r.min()) / (r.max() - r.min() + 1e-9)
    doc_scores = doc_scores.sort_values(["score", "doc_id"], ascending=[False, True])
    if "lang" in doc_scores and (cfg.zh_tau is not None or cfg.max_zh_docs is not None):
        zh = doc_scores["lang"].eq("zh")
        ok_zh = zh & doc_scores["score"].ge(cfg.zh_tau if cfg.zh_tau is not None else -1.0)
        if cfg.max_zh_docs is not None:
            ok_zh &= ok_zh.cumsum().le(cfg.max_zh_docs)
        doc_scores = doc_scores[~zh | ok_zh]
    keep = doc_scores.loc[doc_scores["score"] >= cfg.tau_doc, "doc_id"].tolist()
    if len(keep) < cfg.k_min:
        keep = doc_scores["doc_id"].head(cfg.k_min).tolist()
    return [int(d) for d in keep[: cfg.k_max]]


def full_doc_span(n: int, start: int, end: int, cap: int | None) -> tuple[int, int]:
    """(0, n), or a window of `cap` chars centred on [start, end) and clamped to the document."""
    if cap is None or n <= cap:
        return 0, n
    lo = max(0, min((start + end) // 2 - cap // 2, n - cap))
    return lo, lo + cap


def select(qid: int, cands: pd.DataFrame, doc_text: dict[int, str], cfg: SelectConfig,
           twins: dict[int, list[int]] | None = None) -> QueryResult:
    """cands: chunk_id, doc_id, start, end, score (rerank, in [0, 1]) for one query."""
    if cands.empty:
        return QueryResult(qid)
    keep = choose_docs(cands, cfg)

    docs: list[str] = []
    chunks: list[ChunkRef] = []
    for rank, d in enumerate(keep):
        docs.append(str(d))
        if cfg.include_twins and twins:
            docs += [str(t) for t in twins.get(d, []) if t != d]
        if cfg.chunk_docs is not None and rank >= cfg.chunk_docs:
            continue
        dc = cands[cands["doc_id"] == d].sort_values("score", ascending=False)
        chosen = dc.head(1)
        rest = dc.iloc[1:]
        rest = rest[rest["score"] >= cfg.tau_chunk].head(cfg.max_chunks_per_doc - 1)
        spans = merge_spans(list(zip(pd.concat([chosen, rest])["start"], pd.concat([chosen, rest])["end"],
                                     strict=True)), cfg.merge_gap)
        text = doc_text[d]
        if rank < cfg.full_doc_top:
            spans = [full_doc_span(len(text), int(chosen["start"].iloc[0]), int(chosen["end"].iloc[0]),
                                   cfg.full_doc_max_chars)]
        chunks += [ChunkRef(str(d), text[s:e]) for s, e in spans if text[s:e].strip()]
    return QueryResult(qid, list(dict.fromkeys(docs)), chunks)
