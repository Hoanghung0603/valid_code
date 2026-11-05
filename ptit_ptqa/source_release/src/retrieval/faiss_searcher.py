"""High-performance FAISS Vector and Sparse Lexical Hybrid Searcher."""

import gc
import glob
import logging
import os
import time
from typing import Any, Dict, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp
import faiss

from src.retrieval.base import Candidate
from src.retrieval.fusion import BaseFusionStrategy, LinearFusionStrategy, RRFFusionStrategy

logger = logging.getLogger(__name__)


class FAISSSearcher:
    """Executes ultra-fast dense and hybrid candidate retrieval using memory-mapped FAISS index."""

    def __init__(
        self,
        index_dir: str,
        cand_pool: int = 150,
        cand_buffer: int = 400
    ):
        self.index_dir = index_dir
        self.cand_pool = cand_pool
        self.cand_buffer = max(cand_buffer, 400)

        self.faiss_bin = os.path.join(index_dir, "faiss_index_flat_ip.bin")
        self.ids_file = os.path.join(index_dir, "chunk_ids.npy")
        self.meta_file = os.path.join(index_dir, "chunks_metadata.parquet")
        self.sparse_file = os.path.join(index_dir, "sparse_csr.npz")
        self.text_dir = os.path.join(index_dir, "text")

        self._index = None
        self._sparse_matrix = None
        self._chunk_ids = None
        self._doc_ids = None
        self._starts = None
        self._ends = None

    def _ensure_loaded(self) -> None:
        """Loads FAISS memory-mapped index, sparse CSR matrix, and chunk metadata arrays."""
        if self._index is not None:
            return

        t0 = time.time()
        logger.info(f"Loading FAISS memory-mapped index from {self.faiss_bin}...")
        self._index = faiss.read_index(self.faiss_bin, faiss.IO_FLAG_MMAP)

        logger.info(f"Loading chunk IDs from {self.ids_file}...")
        self._chunk_ids = np.load(self.ids_file)

        logger.info(f"Loading chunk metadata arrays from {self.meta_file}...")
        df_meta = pd.read_parquet(self.meta_file)
        self._doc_ids = df_meta["doc_id"].values
        self._starts = df_meta["start"].values
        self._ends = df_meta["end"].values
        del df_meta

        if os.path.exists(self.sparse_file):
            logger.info(f"Loading Sparse CSR matrix from {self.sparse_file}...")
            self._sparse_matrix = sp.load_npz(self.sparse_file)
        else:
            logger.warning(f"Sparse file {self.sparse_file} not found; hybrid will fall back to dense only.")
            self._sparse_matrix = None

        logger.info(f"FAISSSearcher artifacts loaded in {time.time() - t0:.2f}s.")

    def _load_texts_for_docs(self, needed_doc_ids: Set[int]) -> Dict[int, str]:
        """Lazy-loads text only for candidate documents to minimize RAM consumption."""
        t0 = time.time()
        texts_map: Dict[int, str] = {}
        p_files = sorted(glob.glob(os.path.join(self.text_dir, "part_*.parquet")))
        
        for p in p_files:
            try:
                df_t = pd.read_parquet(p, columns=["doc_id", "text"])
                subset = df_t[df_t["doc_id"].isin(needed_doc_ids)]
                for row in subset.itertuples(index=False):
                    texts_map[int(row.doc_id)] = str(row.text)
                if len(texts_map) >= len(needed_doc_ids):
                    break
            except Exception as e:
                logger.warning(f"Error reading text shard {p}: {e}")

        logger.info(f"Lazy loaded {len(texts_map):,}/{len(needed_doc_ids):,} candidate doc texts in {time.time() - t0:.2f}s.")
        return texts_map

    def search_linear(
        self,
        shards: Any,
        q_vecs: Optional[np.ndarray],
        q_sparse: Optional[sp.csr_matrix],
        num_queries: int,
        configs: List[Dict[str, Any]],
        mode: str = "hybrid",
        valid_doc_ids: Optional[Set[int]] = None,
        fusion_strategy: Optional[LinearFusionStrategy] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes ultra-fast FAISS candidate search followed by sparse sub-matrix rescoring.
        """
        self._ensure_loaded()
        t_start = time.time()
        logger.info(f"🚀 Running FAISS Search (Mode: {mode.upper()}) for {num_queries} queries...")

        cand_k = min(self.cand_buffer, self._index.ntotal)
        
        # 1. Normalize query vectors and search FAISS
        dense_q = q_vecs.astype(np.float32) if q_vecs is not None else None
        if dense_q is not None:
            faiss.normalize_L2(dense_q)
            t_faiss = time.time()
            scores_dense, top_indices = self._index.search(dense_q, cand_k)
            logger.info(f"FAISS Dense Search ({num_queries} queries, top {cand_k}): {time.time() - t_faiss:.2f}s.")
        else:
            raise ValueError("FAISSSearcher requires dense vectors for candidate generation.")

        # Initialize results structure
        for cfg in configs:
            cfg["results"] = [[] for _ in range(num_queries)]

        # 2. Rescore and rank candidates
        needed_doc_ids: Set[int] = set()
        staged_results: Dict[int, List[List[Tuple[float, int, int, int, float]]]] = {
            c_idx: [[] for _ in range(num_queries)] for c_idx in range(len(configs))
        }

        has_sparse = mode in ["hybrid", "sparse"] and q_sparse is not None and self._sparse_matrix is not None
        t_rescore = time.time()

        for q_idx in range(num_queries):
            c_indices = top_indices[q_idx]
            d_scores = scores_dense[q_idx]

            s_scores = None
            if has_sparse:
                sub_sparse = self._sparse_matrix[c_indices]
                s_scores = sub_sparse.dot(q_sparse[q_idx].T).toarray().ravel()

            for c_idx, cfg in enumerate(configs):
                dw = cfg.get("dense_w", 1.0)
                sw = cfg.get("sparse_w", 1.0)

                if mode == "dense" or not has_sparse:
                    q_scores = d_scores
                elif mode == "sparse":
                    q_scores = s_scores
                else:
                    q_scores = dw * d_scores + sw * s_scores

                rank_order = np.argsort(q_scores)[::-1][:self.cand_pool]
                for r in rank_order:
                    row_idx = c_indices[r]
                    d_id = int(self._doc_ids[row_idx])
                    if valid_doc_ids is not None and d_id not in valid_doc_ids:
                        continue

                    needed_doc_ids.add(d_id)
                    score = float(q_scores[r])
                    d_score = float(d_scores[r])
                    staged_results[c_idx][q_idx].append(
                        (score, d_id, int(self._starts[row_idx]), int(self._ends[row_idx]), d_score)
                    )

        logger.info(f"Sparse Rescoring & candidate staging finished in {time.time() - t_rescore:.2f}s.")

        # 3. Lazy text loading for extracted candidates
        texts_map = self._load_texts_for_docs(needed_doc_ids)

        # 4. Materialize final candidate tuples
        for c_idx, cfg in enumerate(configs):
            for q_idx in range(num_queries):
                for score, doc_id, start, end, d_score in staged_results[c_idx][q_idx]:
                    full_text = texts_map.get(doc_id)
                    chunk_text = full_text[start:end] if full_text else ""
                    cfg["results"][q_idx].append((score, doc_id, chunk_text, d_score))

        del texts_map, staged_results
        gc.collect()
        logger.info(f"Total FAISSSearcher execution completed in {time.time() - t_start:.2f}s!")
        return configs

    def search_rrf(
        self,
        shards: Any,
        q_vecs: Optional[np.ndarray],
        q_sparse: Optional[sp.csr_matrix],
        num_queries: int,
        valid_doc_ids: Optional[Set[int]] = None,
        fusion_strategy: Optional[RRFFusionStrategy] = None
    ) -> Tuple[List[List[Tuple[float, int, int, str]]], List[List[Tuple[float, int, int, str]]]]:
        """
        Collects dense and sparse pools using FAISS and Sparse matrix for RRF fusion.
        """
        self._ensure_loaded()
        t_start = time.time()
        logger.info(f"🚀 Running FAISS RRF Search for {num_queries} queries...")

        cand_k = min(self.cand_buffer, self._index.ntotal)
        dense_cand_pool: List[List[Tuple[float, int, int, str]]] = [[] for _ in range(num_queries)]
        sparse_cand_pool: List[List[Tuple[float, int, int, str]]] = [[] for _ in range(num_queries)]

        dense_q = q_vecs.astype(np.float32) if q_vecs is not None else None
        if dense_q is not None:
            faiss.normalize_L2(dense_q)
            scores_dense, top_indices = self._index.search(dense_q, cand_k)
        else:
            raise ValueError("FAISSSearcher requires dense vectors.")

        needed_doc_ids: Set[int] = set()
        staged_dense: List[List[Tuple[float, int, int, int, int]]] = [[] for _ in range(num_queries)]
        staged_sparse: List[List[Tuple[float, int, int, int, int]]] = [[] for _ in range(num_queries)]

        has_sparse = q_sparse is not None and self._sparse_matrix is not None

        for q_idx in range(num_queries):
            c_indices = top_indices[q_idx]
            d_scores = scores_dense[q_idx]

            # Dense candidates
            d_order = np.argsort(d_scores)[::-1][:self.cand_pool]
            for r in d_order:
                row_idx = c_indices[r]
                d_id = int(self._doc_ids[row_idx])
                cid = int(self._chunk_ids[row_idx])
                if valid_doc_ids is not None and d_id not in valid_doc_ids:
                    continue
                needed_doc_ids.add(d_id)
                staged_dense[q_idx].append((float(d_scores[r]), cid, d_id, int(self._starts[row_idx]), int(self._ends[row_idx])))

            # Sparse candidates
            if has_sparse:
                sub_sparse = self._sparse_matrix[c_indices]
                s_scores = sub_sparse.dot(q_sparse[q_idx].T).toarray().ravel()
                s_order = np.argsort(s_scores)[::-1][:self.cand_pool]
                for r in s_order:
                    row_idx = c_indices[r]
                    d_id = int(self._doc_ids[row_idx])
                    cid = int(self._chunk_ids[row_idx])
                    if valid_doc_ids is not None and d_id not in valid_doc_ids:
                        continue
                    needed_doc_ids.add(d_id)
                    staged_sparse[q_idx].append((float(s_scores[r]), cid, d_id, int(self._starts[row_idx]), int(self._ends[row_idx])))

        texts_map = self._load_texts_for_docs(needed_doc_ids)

        for q_idx in range(num_queries):
            for score, cid, doc_id, start, end in staged_dense[q_idx]:
                txt = texts_map.get(doc_id, "")[start:end] if doc_id in texts_map else ""
                dense_cand_pool[q_idx].append((score, cid, doc_id, txt))

            for score, cid, doc_id, start, end in staged_sparse[q_idx]:
                txt = texts_map.get(doc_id, "")[start:end] if doc_id in texts_map else ""
                sparse_cand_pool[q_idx].append((score, cid, doc_id, txt))

        del texts_map, staged_dense, staged_sparse
        gc.collect()
        logger.info(f"FAISS RRF Search completed in {time.time() - t_start:.2f}s.")
        return dense_cand_pool, sparse_cand_pool
