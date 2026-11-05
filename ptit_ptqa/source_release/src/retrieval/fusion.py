"""Fusion strategies for combining dense and sparse retrieval modalities."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np

from src.retrieval.base import Candidate


class BaseFusionStrategy(ABC):
    """Abstract Strategy interface for multi-modal candidate fusion and pruning."""

    @abstractmethod
    def prune_candidates(
        self,
        pool: List[Any],
        cand_limit: int
    ) -> List[Any]:
        """Prunes candidate pool to maintain memory efficiency across shards."""
        pass

    @abstractmethod
    def fuse(
        self,
        dense_candidates: List[Tuple[float, int, int, str]],  # (score, chunk_id, doc_id, chunk_text)
        sparse_candidates: List[Tuple[float, int, int, str]],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        **kwargs: Any
    ) -> List[Candidate]:
        """Combines multimodal candidates into a unified list of Candidates."""
        pass


class RRFFusionStrategy(BaseFusionStrategy):
    """Reciprocal Rank Fusion (RRF): score = w_d / (k + r_d) + w_s / (k + r_s)."""

    def __init__(self, rrf_k: int = 60):
        self.rrf_k = rrf_k

    def prune_candidates(
        self,
        pool: List[Tuple[float, int, int, str]],
        cand_limit: int
    ) -> List[Tuple[float, int, int, str]]:
        if len(pool) > cand_limit:
            return sorted(pool, key=lambda x: x[0], reverse=True)[:cand_limit]
        return pool

    def fuse(
        self,
        dense_candidates: List[Tuple[float, int, int, str]],
        sparse_candidates: List[Tuple[float, int, int, str]],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        **kwargs: Any
    ) -> List[Candidate]:
        # 1. Deduplicate & rank dense pool
        dense_best: Dict[int, float] = {}
        meta_lookup: Dict[int, Tuple[int, str]] = {}
        for sc, cid, did, ctxt in dense_candidates:
            if cid not in dense_best or sc > dense_best[cid]:
                dense_best[cid] = sc
                meta_lookup[cid] = (did, ctxt)

        dense_sorted = sorted(dense_best.items(), key=lambda x: x[1], reverse=True)
        dense_rank = {cid: rank + 1 for rank, (cid, _) in enumerate(dense_sorted)}

        # 2. Deduplicate & rank sparse pool
        sparse_best: Dict[int, float] = {}
        for sc, cid, did, ctxt in sparse_candidates:
            if cid not in sparse_best or sc > sparse_best[cid]:
                sparse_best[cid] = sc
                if cid not in meta_lookup:
                    meta_lookup[cid] = (did, ctxt)

        sparse_sorted = sorted(sparse_best.items(), key=lambda x: x[1], reverse=True)
        sparse_rank = {cid: rank + 1 for rank, (cid, _) in enumerate(sparse_sorted)}

        # 3. Calculate reciprocal rank scores
        all_cids = set(dense_rank.keys()) | set(sparse_rank.keys())
        results: List[Candidate] = []
        for cid in all_cids:
            score = 0.0
            if cid in dense_rank:
                score += dense_weight / (self.rrf_k + dense_rank[cid])
            if cid in sparse_rank:
                score += sparse_weight / (self.rrf_k + sparse_rank[cid])

            did, ctxt = meta_lookup[cid]
            d_score = dense_best.get(cid)
            results.append(Candidate(
                score=score,
                doc_id=did,
                chunk_text=ctxt,
                dense_score=d_score,
                chunk_id=cid
            ))

        results.sort(key=lambda x: x.score, reverse=True)
        return results


class LinearFusionStrategy(BaseFusionStrategy):
    """Linear Weighted Score Fusion with Document-level diversity pruning."""

    def prune_candidates(
        self,
        pool: List[Tuple[float, int, str, float]],  # (score, doc_id, chunk_text, dense_score)
        cand_limit: int
    ) -> List[Tuple[float, int, str, float]]:
        if len(pool) <= cand_limit:
            return pool

        sorted_items = sorted(pool, key=lambda x: x[0], reverse=True)
        pruned: List[Tuple[float, int, str, float]] = []
        doc_chunk_cnt: Dict[int, int] = {}

        # First pass: limit each document to at most 2 chunks for diversity
        for item in sorted_items:
            did = item[1]
            c_cnt = doc_chunk_cnt.get(did, 0)
            if c_cnt < 2:
                pruned.append(item)
                doc_chunk_cnt[did] = c_cnt + 1
                if len(pruned) >= cand_limit:
                    break

        # Second pass: fill remaining capacity if needed
        if len(pruned) < cand_limit:
            pruned_texts = {c[2] for c in pruned}
            for item in sorted_items:
                if len(pruned) >= cand_limit:
                    break
                if item[2] not in pruned_texts:
                    pruned.append(item)
                    pruned_texts.add(item[2])

        return pruned

    def fuse(
        self,
        dense_candidates: List[Any],
        sparse_candidates: List[Any],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        raw_items: Optional[List[Tuple[float, int, str, float]]] = None,
        **kwargs: Any
    ) -> List[Candidate]:
        items = raw_items or []
        candidates: List[Candidate] = []
        for item in items:
            score = float(item[0])
            doc_id = int(item[1])
            chunk_text = str(item[2])
            dense_score = float(item[3]) if len(item) > 3 else score
            candidates.append(Candidate(
                score=score,
                doc_id=doc_id,
                chunk_text=chunk_text,
                dense_score=dense_score
            ))

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates
