"""Pipeline orchestrator for end-to-end retrieval, fusion, reranking, and submission."""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from src.retrieval.base import BaseQueryEncoder, Candidate, QueryBundle, RetrievalResult
from src.retrieval.fusion import BaseFusionStrategy, LinearFusionStrategy, RRFFusionStrategy
from src.retrieval.index_repository import IndexRepository, IndexShard
from src.retrieval.searcher import VectorSearcher
from src.reranking.base import BaseReranker
from src.pipeline.submission_exporter import SubmissionExporter

logger = logging.getLogger(__name__)


class SubmissionPipeline:
    """Orchestrates query encoding, index search, fusion, reranking, and export."""

    def __init__(
        self,
        encoder: BaseQueryEncoder,
        searcher: Any,
        index_repository: IndexRepository,
        exporter: SubmissionExporter,
        reranker: Optional[BaseReranker] = None
    ):
        self.encoder = encoder
        self.searcher = searcher
        self.index_repo = index_repository
        self.exporter = exporter
        self.reranker = reranker

    @staticmethod
    def load_queries(query_path: str, enable_multilingual: bool = True) -> List[QueryBundle]:
        """Loads search queries from parquet and auto-merges translated Chinese and English if present and enabled."""
        df_q = pd.read_parquet(query_path)

        if enable_multilingual and os.path.exists("data/raw/btc_provided/query_translated.parquet"):
            try:
                df_trans = pd.read_parquet("data/raw/btc_provided/query_translated.parquet")
                merge_cols = [c for c in ["query_zh", "query_en"] if c in df_trans.columns and c not in df_q.columns]
                if merge_cols and set(df_q["id"]).issubset(set(df_trans["id"])):
                    df_q = df_q.merge(df_trans[["id"] + merge_cols], on="id", how="left")
                    logger.info(f"Auto-merged multilingual translations {merge_cols} for retrieval.")
            except Exception as e:
                logger.warning(f"Failed to merge translations: {e}")

        queries: List[QueryBundle] = []
        has_zh = "query_zh" in df_q.columns and (df_q["query_zh"] != "").any()
        has_en = "query_en" in df_q.columns and (df_q["query_en"] != "").any()

        for row in df_q.itertuples(index=False):
            q_id = int(row.id)
            q_vi = str(getattr(row, "query", ""))
            q_zh = str(getattr(row, "query_zh", "")) if has_zh else None
            q_en = str(getattr(row, "query_en", "")) if has_en else None
            queries.append(QueryBundle(id=q_id, text_vi=q_vi, text_zh=q_zh, text_en=q_en))

        logger.info(
            f"Loaded {len(queries)} queries. Multilingual: "
            f"Chinese={'ENABLED' if has_zh else 'DISABLED'}, English={'ENABLED' if has_en else 'DISABLED'}"
        )
        return queries

    def execute(
        self,
        queries: List[QueryBundle],
        configs: List[Dict[str, Any]],
        mode: str = "hybrid",
        fusion_type: str = "rrf",
        zh_weight: float = 1.0,
        en_weight: float = 0.5,
        rrf_k: int = 60,
        top_docs: int = 100,
        top_chunks: int = 98,
        score_threshold: Optional[float] = None,
        threshold_metric: str = "dense",
        min_docs: int = 5,
        min_chunks: int = 5,
        valid_doc_ids: Optional[Set[int]] = None,
        rerank_batch_size: int = 64
    ) -> List[Dict[str, Any]]:
        """
        Runs the complete retrieval and submission generation workflow.
        """
        num_queries = len(queries)
        queries_vi = [q.text_vi for q in queries]
        has_zh = any(bool(q.text_zh and q.text_zh.strip()) for q in queries)
        queries_zh = [q.text_zh or "" for q in queries] if has_zh else None
        has_en = any(bool(q.text_en and q.text_en.strip()) for q in queries)
        queries_en = [q.text_en or "" for q in queries] if has_en else None

        # 1. Encode queries
        logger.info(f"=== Encoding queries (Mode: {mode}) ===")
        t_enc = time.time()
        need_dense = mode in ["dense", "hybrid"]
        need_sparse = mode in ["sparse", "hybrid"]

        q_vecs = None
        q_sparse = None

        cache_file = os.path.join("data", "indices", "new_dataset", ".cache", "query_encodings_1200.pkl")
        loaded_from_cache = False

        if os.path.exists(cache_file) and len(queries) == 1200:
            try:
                import pickle
                logger.info(f"⚡ Found cached trilingual query encodings at {cache_file}! Loading instantly...")
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)

                if need_dense:
                    q_vecs = cached_data["dense_q"]

                if need_sparse:
                    fused_lexical = []
                    for i in range(len(queries)):
                        merged = dict(cached_data["lex_vi"][i])
                        if has_zh and "lex_zh" in cached_data:
                            for tok, val in cached_data["lex_zh"][i].items():
                                merged[tok] = max(merged.get(tok, 0.0), float(val) * zh_weight)
                        if has_en and "lex_en" in cached_data:
                            for tok, val in cached_data["lex_en"][i].items():
                                merged[tok] = max(merged.get(tok, 0.0), float(val) * en_weight)
                        fused_lexical.append(merged)
                    q_sparse = self.encoder.build_query_sparse_matrix(fused_lexical)

                loaded_from_cache = True
                logger.info("Successfully loaded query encodings from cache without loading FlagEmbedding/PyTorch into memory!")
            except Exception as e:
                logger.warning(f"Failed to load cached query encodings: {e}. Falling back to dynamic encoding...")

        if not loaded_from_cache:
            if need_dense:
                logger.info("Encoding Dense vectors (Vietnamese)...")
                q_vecs = self.encoder.encode_dense(queries_vi)

            if need_sparse:
                logger.info(f"Encoding Sparse lexical vectors (Multilingual zh_w={zh_weight}, en_w={en_weight})...")
                q_sparse = self.encoder.encode_sparse(
                    queries_vi=queries_vi,
                    queries_zh=queries_zh,
                    queries_en=queries_en,
                    zh_weight=zh_weight,
                    en_weight=en_weight
                )

            # Unload encoder to release GPU memory for subsequent reranker
            self.encoder.unload()

        logger.info(f"Query encoding finished in {time.time() - t_enc:.2f}s.")

        # 2. Discover shards
        shards = self.index_repo.discover_shards()
        logger.info(f"Discovered {len(shards)} unique index shards.")

        # 3. Execute Search
        fusion_mode = fusion_type if mode == "hybrid" else "linear"
        dense_pool = None
        sparse_pool = None

        if fusion_mode == "rrf":
            rrf_strat = RRFFusionStrategy(rrf_k=rrf_k)
            dense_pool, sparse_pool = self.searcher.search_rrf(
                shards=shards,
                q_vecs=q_vecs,
                q_sparse=q_sparse,
                num_queries=num_queries,
                valid_doc_ids=valid_doc_ids,
                fusion_strategy=rrf_strat
            )
        else:
            linear_strat = LinearFusionStrategy()
            configs = self.searcher.search_linear(
                shards=shards,
                q_vecs=q_vecs,
                q_sparse=q_sparse,
                num_queries=num_queries,
                configs=configs,
                mode=mode,
                valid_doc_ids=valid_doc_ids,
                fusion_strategy=linear_strat
            )

        # 4. Process each configuration
        rrf_strat = RRFFusionStrategy(rrf_k=rrf_k)
        linear_strat = LinearFusionStrategy()

        for c_idx, cfg in enumerate(configs):
            tag = cfg.get("tag", f"cfg_{c_idx}")
            dw = cfg.get("dense_w", 1.0)
            sw = cfg.get("sparse_w", 1.0)
            th_val = cfg.get("threshold", score_threshold)
            logger.info(f"\nProcessing Config [{c_idx+1}/{len(configs)}]: Tag={tag}, Dense_w={dw:.2f}, Sparse_w={sw:.2f}")

            results: List[RetrievalResult] = []
            t_cfg_start = time.time()

            for q_idx in range(num_queries):
                q_bundle = queries[q_idx]

                # Fuse candidates
                if fusion_mode == "rrf" and dense_pool is not None and sparse_pool is not None:
                    candidates = rrf_strat.fuse(
                        dense_candidates=dense_pool[q_idx],
                        sparse_candidates=sparse_pool[q_idx],
                        dense_weight=dw,
                        sparse_weight=sw
                    )
                else:
                    candidates = linear_strat.fuse(
                        dense_candidates=[],
                        sparse_candidates=[],
                        raw_items=cfg["results"][q_idx]
                    )

                # Stage 2: Rerank OR Stage 1 Filter
                if self.reranker is not None:
                    rel_docs, rel_chunks = self.reranker.rerank(
                        query=q_bundle.text_vi,
                        candidates=candidates,
                        query_zh=q_bundle.text_zh,
                        top_docs=top_docs,
                        top_chunks=top_chunks,
                        batch_size=rerank_batch_size
                    )
                else:
                    seen_docs: Set[int] = set()
                    seen_chunk_texts: Set[str] = set()
                    rel_docs = []
                    rel_chunks = []

                    for cand in candidates:
                        metric_val = cand.dense_score if threshold_metric == "dense" else cand.score

                        # Document threshold filtering
                        if th_val is not None:
                            if len(rel_docs) >= min_docs and metric_val < th_val:
                                pass
                            elif cand.doc_id not in seen_docs and len(rel_docs) < top_docs:
                                seen_docs.add(cand.doc_id)
                                rel_docs.append(int(cand.doc_id))
                        else:
                            if cand.doc_id not in seen_docs and len(rel_docs) < top_docs:
                                seen_docs.add(cand.doc_id)
                                rel_docs.append(int(cand.doc_id))

                        if not cand.chunk_text or not cand.chunk_text.strip():
                            continue

                        # Chunk threshold filtering
                        if th_val is not None:
                            if len(rel_chunks) >= min_chunks and metric_val < th_val:
                                pass
                            elif cand.chunk_text not in seen_chunk_texts and len(rel_chunks) < top_chunks:
                                seen_chunk_texts.add(cand.chunk_text)
                                rel_chunks.append({
                                    "doc_id": int(cand.doc_id),
                                    "chunk_text": cand.chunk_text
                                })
                        else:
                            if cand.chunk_text not in seen_chunk_texts and len(rel_chunks) < top_chunks:
                                seen_chunk_texts.add(cand.chunk_text)
                                rel_chunks.append({
                                    "doc_id": int(cand.doc_id),
                                    "chunk_text": cand.chunk_text
                                })

                        if len(rel_docs) == top_docs and len(rel_chunks) == top_chunks:
                            break

                results.append(RetrievalResult(
                    query_id=q_bundle.id,
                    relevant_docs=rel_docs[:top_docs],
                    relevant_chunks=rel_chunks[:top_chunks]
                ))

                if (q_idx + 1) % 100 == 0 or (q_idx + 1) == num_queries:
                    rate = (q_idx + 1) / max(time.time() - t_cfg_start, 0.001)
                    logger.info(f"Progress: {q_idx + 1}/{num_queries} queries ({rate:.1f} q/s)...")

            # 5. Export results
            self.exporter.export(
                results=results,
                out_json=cfg["out_json"],
                out_zip=cfg["out_zip"],
                initial_top_chunks=top_chunks
            )

        return configs
