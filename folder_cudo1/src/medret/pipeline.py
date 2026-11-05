"""End-to-end run: queries -> translate -> encode -> retrieve -> rerank -> select -> submission (data/runs/<run>/).

Stage outputs follow architecture.md section 3: config.yaml, queries.parquet (translations), candidates.parquet,
reranked.parquet, submission.json and the zip. Every model call is cached (translations, query vectors, rerank
scores), so re-running with other selection thresholds costs seconds.
"""

from __future__ import annotations

import gc
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from medret.config import PROJECT_ROOT, Config
from medret.retrieve.pipeline import RetrieveConfig
from medret.select.selector import SelectConfig

RERANK_QUERY_BATCH = 100  # queries whose chunk texts are held in memory at once during rerank


@dataclass
class RerankConfig:
    top_m: int = 100  # RRF candidates per query sent to the reranker
    batch_docs: int = 32
    threads: int = 8
    # "auto": each chunk is scored against the query in its own language (zh chunks with the translation);
    # scoring zh chunks with the Vietnamese query ranked them far below vi chunks on the dev set (MRR 0.08)
    query_side: str = "auto"


@dataclass
class RunConfig:
    retrieve: RetrieveConfig = field(default_factory=RetrieveConfig)
    rerank: RerankConfig = field(default_factory=RerankConfig)
    select: SelectConfig = field(default_factory=SelectConfig)
    translate_model: str = "Qwen3.6-27B"
    allow_dev_models: bool = False


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "uncommitted"


def run_pipeline(cfg: Config, run_name: str, queries: pd.DataFrame, rc: RunConfig, log=print,
                 write_zip: bool = True) -> Path:
    from medret.index.embed_input import passage_input
    from medret.rerank.fpt_api import FPTReranker
    from medret.retrieve.pipeline import Retriever
    from medret.retrieve.query_encoder import QueryEncoder
    from medret.retrieve.translate import DEV_ONLY, PROMPT_VERSION_EN, SYSTEM_EN, Translator, assert_submittable
    from medret.submit import pack_submission, validate_submission

    out = cfg.paths.runs_dir / run_name
    out.mkdir(parents=True, exist_ok=True)
    cache = cfg.paths.data_dir / "cache"
    (out / "config.yaml").write_text(yaml.safe_dump({
        "run": run_name, "commit": _git_commit(), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_queries": len(queries), "embed_model": cfg.embed.model, "dense_run": cfg.index.dense_run,
        "translate_model": rc.translate_model, "translate_dev_only": rc.translate_model in DEV_ONLY,
        "translation_used": any(ch.endswith(("_zh", "_en")) for ch in rc.retrieve.channels) or rc.rerank.query_side == "zh",
        "reranker": "bge-reranker-v2-m3 (FPT API)", **asdict(rc)}, allow_unicode=True, sort_keys=False),
        encoding="utf-8")

    t0 = time.time()
    q = queries[["id", "query"]].reset_index(drop=True).copy()
    # translate only when a channel or the reranker reads the translation: until 2026-10-31 query_zh was always made
    # (Qwen3.6-27B, dev only), and with query_side "auto" the zh chunks that the vi channels found were reranked
    # against it (none of those scores reaches the 0.3735 submission; vi-only runs now rerank them with the vi query)
    use_zh = any(ch.endswith("_zh") for ch in rc.retrieve.channels) or rc.rerank.query_side == "zh"
    use_en = any(ch.endswith("_en") for ch in rc.retrieve.channels)
    if (use_zh or use_en) and not rc.allow_dev_models:
        assert_submittable([rc.translate_model])
    q["query_zh"] = None
    if use_zh:
        q["query_zh"] = Translator(rc.translate_model, cache_path=cache / "translate.sqlite",
                                   threads=cfg.llm.threads).translate(q["query"].tolist())
    q["query_en"] = None
    if use_en:
        q["query_en"] = Translator(rc.translate_model, cache_path=cache / "translate.sqlite", threads=cfg.llm.threads,
                                   system=SYSTEM_EN, prompt_version=PROMPT_VERSION_EN).translate(q["query"].tolist())
    q.to_parquet(out / "queries.parquet", index=False)
    enc = QueryEncoder(cfg.embed.model, cache_path=cache / "qvec.sqlite")
    v_vi = enc.encode(q["query"].tolist())
    v_zh = enc.encode(q["query_zh"].tolist()) if use_zh else [None] * len(q)
    v_en = enc.encode(q["query_en"].tolist()) if use_en else [None] * len(q)
    log(f"[run] translate + encode: {time.time() - t0:.0f}s")

    t0 = time.time()
    retriever = Retriever(cfg.paths.index_dir, cfg.index.dense_run, cfg.index.nprobe, cfg.index.k_coarse,
                          rc.retrieve, log=log)
    parts = []
    for i, r in enumerate(q.itertuples()):
        c = retriever.retrieve(r.query, r.query_zh, v_vi[i], v_zh[i], r.query_en, v_en[i])
        parts.append(c.assign(query_id=r.id))
        if (i + 1) % 200 == 0:
            log(f"[run] retrieved {i + 1}/{len(q)} queries")
    cands = pd.concat(parts, ignore_index=True)
    cands.to_parquet(out / "candidates.parquet", index=False)
    # reranking needs only the metadata store: release FAISS, BM25 and the vector memmap pages touched by the
    # exact rescoring (~GBs resident after 1,200 queries; the laptop's RAM guard stopped the 2026-10-20 run)
    meta_store = retriever.meta
    del retriever, parts
    gc.collect()
    log(f"[run] retrieve: {len(cands):,} candidates in {time.time() - t0:.0f}s, routes per channel "
        f"{ {ch: int(cands['routes'].str.contains(ch).sum()) for ch in rc.retrieve.channels} }")

    t0 = time.time()
    top = cands.sort_values(["query_id", "rrf"], ascending=[True, False]).groupby("query_id").head(rc.rerank.top_m)
    q_vi, q_zh = dict(zip(q["id"], q["query"], strict=True)), dict(zip(q["id"], q["query_zh"], strict=True))
    q_en = dict(zip(q["id"], q["query_en"], strict=True))
    side = rc.rerank.query_side

    def rq(qi: int, lang: str) -> str:
        """Query text paired with a chunk: same language as the chunk when side == "auto"."""
        if side == "zh" or (side == "auto" and lang == "zh" and q_zh[qi]):
            return q_zh[qi]
        if side == "auto" and lang == "en" and q_en[qi]:
            return q_en[qi]
        return q_vi[qi]

    rr = FPTReranker(cache_path=cache / "rerank.sqlite", batch_docs=rc.rerank.batch_docs, threads=rc.rerank.threads)
    # chunk texts are loaded per batch of queries: all of them at once does not fit in RAM for deep pools
    qids, done, n_pairs = top["query_id"].unique(), [], 0
    for s in range(0, len(qids), RERANK_QUERY_BATCH):
        part = top[top["query_id"].isin(qids[s: s + RERANK_QUERY_BATCH])]
        meta = meta_store.chunks(part["chunk_id"].unique()).set_index("chunk_id")
        part = part.join(meta[["doc_id", "lang", "start", "end", "title", "heading_path", "chunk_text"]],
                         on="chunk_id")
        pairs = [(rq(qi, lg), passage_input(t, h, ct)) for qi, lg, t, h, ct in
                 zip(part["query_id"], part["lang"], part["title"], part["heading_path"], part["chunk_text"],
                     strict=True)]
        done.append(part.assign(score=rr.score_pairs(pairs)).drop(columns=["chunk_text"]))
        n_pairs += len(pairs)
        if len(qids) > RERANK_QUERY_BATCH:
            log(f"[run] reranked {min(s + RERANK_QUERY_BATCH, len(qids))}/{len(qids)} queries")
    reranked = pd.concat(done, ignore_index=True)
    reranked.to_parquet(out / "reranked.parquet", index=False)
    log(f"[run] rerank: {n_pairs:,} pairs ({rr.stats.cached:,} cached, {rr.stats.requests} requests, "
        f"{rr.stats.tokens:,} tokens) in {time.time() - t0:.0f}s")

    sub_path = select_and_write(cfg, out, q["id"].tolist(), reranked, rc.select, meta_store, log=log)
    errors = validate_submission(sub_path, q["id"].tolist())
    log(f"[run] validate: {'OK' if not errors else errors[:5]}")
    if write_zip and not errors:  # zip named after the run, e.g. data/runs/sub-vi200-v1/sub-vi200-v1.zip
        log(f"[run] zip: {pack_submission(sub_path, out / f'{run_name}.zip')}")
    return sub_path


def select_and_write(cfg: Config, out: Path, query_ids: list[int], reranked: pd.DataFrame, sc: SelectConfig,
                     meta, log=print) -> Path:
    """Selection only (cheap): reused by `medret select` to try thresholds on a finished run."""
    from medret.select.selector import choose_docs, select
    from medret.submit import write_submission

    by_q = {qid: g for qid, g in reranked.groupby("query_id")}
    picked_docs = {d for g in by_q.values() for d in choose_docs(g, sc)}
    texts = meta.documents(picked_docs, with_text=True).set_index("doc_id")["text"].to_dict()
    twins = None
    if sc.include_twins:
        tw = meta.twins(picked_docs)
        twins = tw.groupby("fetched_id")["id"].apply(list).to_dict() if not tw.empty else {}
    results = [select(int(qid), by_q.get(qid, pd.DataFrame()), texts, sc, twins) for qid in query_ids]
    path = write_submission(results, out / "submission.json", cfg.submission.doc_id_type)
    n_docs = np.array([len(r.relevant_docs) for r in results])
    n_chunks = np.array([len(r.relevant_chunks) for r in results])
    (out / "select.json").write_text(json.dumps({**asdict(sc), "docs_per_query_mean": float(n_docs.mean()),
                                                 "chunks_per_query_mean": float(n_chunks.mean())}, indent=1))
    log(f"[run] select: docs/query mean {n_docs.mean():.1f} (min {n_docs.min()}), "
        f"chunks/query mean {n_chunks.mean():.1f} -> {path}")
    return path
