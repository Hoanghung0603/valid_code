"""Vector and lexical search execution engine across distributed index shards."""

import gc
import logging
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import scipy.sparse as sp

from src.retrieval.base import Candidate
from src.retrieval.fusion import BaseFusionStrategy, LinearFusionStrategy, RRFFusionStrategy
from src.retrieval.index_repository import IndexRepository, IndexShard
from src.retrieval.faiss_searcher import FAISSSearcher

logger = logging.getLogger(__name__)


class VectorSearcher:
    """Executes dense and sparse candidate retrieval across index shards."""

    def __init__(
        self,
        index_repository: IndexRepository,
        q_batch_size: int = 100,
        cand_pool: int = 150,
        cand_buffer: int = 400
    ):
        self.index_repo = index_repository
        self.q_batch_size = q_batch_size
        self.cand_pool = cand_pool
        self.cand_buffer = cand_buffer

    def search_rrf(
        self,
        shards: List[IndexShard],
        q_vecs: Optional[np.ndarray],
        q_sparse: Optional[sp.csr_matrix],
        num_queries: int,
        valid_doc_ids: Optional[Set[int]] = None,
        fusion_strategy: Optional[RRFFusionStrategy] = None
    ) -> Tuple[List[List[Tuple[float, int, int, str]]], List[List[Tuple[float, int, int, str]]]]:
        """
        Executes search specifically collecting dense and sparse pools for RRF fusion.
        Returns (dense_cand_pool, sparse_cand_pool).
        """
        strat = fusion_strategy or RRFFusionStrategy()
        dense_cand_pool: List[List[Tuple[float, int, int, str]]] = [[] for _ in range(num_queries)]
        sparse_cand_pool: List[List[Tuple[float, int, int, str]]] = [[] for _ in range(num_queries)]

        need_dense = q_vecs is not None
        need_sparse = q_sparse is not None

        for s_idx, shard in enumerate(shards):
            t_shard = time.time()
            logger.info(f"[{s_idx+1}/{len(shards)}] Searching shard: {shard.name}...")

            chunks_map, total_chunks = shard.load_chunks(valid_doc_ids=valid_doc_ids)
            if not chunks_map:
                logger.warning(f"Shard {shard.name} has no valid documents in whitelist. Skipping.")
                continue

            texts_map = shard.load_texts(valid_doc_ids=valid_doc_ids)

            for c_ids, c_vecs, M in shard.iter_embedding_blocks(need_dense=need_dense, need_sparse=need_sparse):
                num_chunks = len(c_ids)

                for q_start in range(0, num_queries, self.q_batch_size):
                    q_end = min(q_start + self.q_batch_size, num_queries)

                    dense_batch = None
                    if need_dense and c_vecs is not None:
                        dense_batch = q_vecs[q_start:q_end] @ c_vecs.T

                    sparse_batch = None
                    if need_sparse:
                        if M is not None:
                            sparse_batch = (M.dot(q_sparse[q_start:q_end].T)).T.toarray()
                        else:
                            sparse_batch = np.zeros((q_end - q_start, num_chunks), dtype=np.float32)

                    for b_i, q_idx in enumerate(range(q_start, q_end)):
                        if need_dense and dense_batch is not None:
                            d_top = np.argsort(dense_batch[b_i])[::-1][:self.cand_pool]
                            for ti in d_top:
                                chunk_id = int(c_ids[ti])
                                c_info = chunks_map.get(chunk_id)
                                if c_info is None:
                                    continue
                                doc_id, start, end = c_info
                                full_text = texts_map.get(doc_id)
                                if full_text is None:
                                    continue
                                chunk_text = full_text[start:end]
                                dense_cand_pool[q_idx].append(
                                    (float(dense_batch[b_i][ti]), chunk_id, int(doc_id), chunk_text)
                                )

                        if need_sparse and sparse_batch is not None and M is not None:
                            s_top = np.argsort(sparse_batch[b_i])[::-1][:self.cand_pool]
                            for ti in s_top:
                                chunk_id = int(c_ids[ti])
                                c_info = chunks_map.get(chunk_id)
                                if c_info is None:
                                    continue
                                doc_id, start, end = c_info
                                full_text = texts_map.get(doc_id)
                                if full_text is None:
                                    continue
                                chunk_text = full_text[start:end]
                                sparse_cand_pool[q_idx].append(
                                    (float(sparse_batch[b_i][ti]), chunk_id, int(doc_id), chunk_text)
                                )

            # Intermediate pruning after shard
            for q_idx in range(num_queries):
                dense_cand_pool[q_idx] = strat.prune_candidates(dense_cand_pool[q_idx], self.cand_buffer)
                sparse_cand_pool[q_idx] = strat.prune_candidates(sparse_cand_pool[q_idx], self.cand_buffer)

            del chunks_map, texts_map
            gc.collect()
            logger.info(f"Shard {shard.name} completed in {time.time() - t_shard:.2f}s.")

        return dense_cand_pool, sparse_cand_pool

    def search_linear(
        self,
        shards: List[IndexShard],
        q_vecs: Optional[np.ndarray],
        q_sparse: Optional[sp.csr_matrix],
        num_queries: int,
        configs: List[Dict[str, Any]],
        mode: str = "hybrid",
        valid_doc_ids: Optional[Set[int]] = None,
        fusion_strategy: Optional[LinearFusionStrategy] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes row-by-row linear score combination across one or multiple configurations.
        """
        strat = fusion_strategy or LinearFusionStrategy()
        need_dense = mode in ["dense", "hybrid"] and q_vecs is not None
        need_sparse = mode in ["sparse", "hybrid"] and q_sparse is not None

        for cfg in configs:
            cfg["results"] = [[] for _ in range(num_queries)]

        cand_pool_size = max(self.cand_pool, 100)

        for s_idx, shard in enumerate(shards):
            t_shard = time.time()
            logger.info(f"[{s_idx+1}/{len(shards)}] Searching shard: {shard.name} (Linear)...")

            chunks_map, total_chunks = shard.load_chunks(valid_doc_ids=valid_doc_ids)
            if not chunks_map:
                continue

            texts_map = shard.load_texts(valid_doc_ids=valid_doc_ids)

            for c_ids, c_vecs, M in shard.iter_embedding_blocks(need_dense=need_dense, need_sparse=need_sparse):
                num_chunks = len(c_ids)

                for q_start in range(0, num_queries, self.q_batch_size):
                    q_end = min(q_start + self.q_batch_size, num_queries)

                    dense_batch = None
                    if need_dense and c_vecs is not None:
                        dense_batch = q_vecs[q_start:q_end] @ c_vecs.T

                    sparse_batch = None
                    if need_sparse:
                        if M is not None:
                            sparse_batch = (M.dot(q_sparse[q_start:q_end].T)).T.toarray()
                        else:
                            sparse_batch = np.zeros((q_end - q_start, num_chunks), dtype=np.float32)

                    for cfg in configs:
                        dw = cfg.get("dense_w", 1.0)
                        sw = cfg.get("sparse_w", 1.0)

                        for b_i, q_idx in enumerate(range(q_start, q_end)):
                            if mode == "dense" and dense_batch is not None:
                                q_scores = dense_batch[b_i]
                            elif mode == "sparse" and sparse_batch is not None:
                                q_scores = sparse_batch[b_i]
                            else:
                                d_part = dw * dense_batch[b_i] if dense_batch is not None else 0
                                s_part = sw * sparse_batch[b_i] if sparse_batch is not None else 0
                                q_scores = d_part + s_part

                            top_idx = np.argsort(q_scores)[::-1][:cand_pool_size]
                            for ti in top_idx:
                                chunk_id = int(c_ids[ti])
                                c_info = chunks_map.get(chunk_id)
                                if c_info is None:
                                    continue
                                doc_id, start, end = c_info
                                full_text = texts_map.get(doc_id)
                                if full_text is None:
                                    continue
                                chunk_text = full_text[start:end]
                                d_score = float(dense_batch[b_i][ti]) if dense_batch is not None else float(q_scores[ti])
                                cfg["results"][q_idx].append(
                                    (float(q_scores[ti]), int(doc_id), chunk_text, d_score)
                                )

            # Intermediate pruning after shard
            for cfg in configs:
                for q_idx in range(num_queries):
                    cfg["results"][q_idx] = strat.prune_candidates(cfg["results"][q_idx], self.cand_buffer)

            del chunks_map, texts_map
            gc.collect()
            logger.info(f"Shard {shard.name} completed in {time.time() - t_shard:.2f}s.")

        return configs
