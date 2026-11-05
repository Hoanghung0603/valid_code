"""Cross-Encoder reranker using BGE-Reranker-v2-m3."""

import logging
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.base import Candidate
from src.reranking.base import BaseReranker

logger = logging.getLogger(__name__)


class CrossEncoderReranker(BaseReranker):
    """Encapsulates BAAI/bge-reranker-v2-m3 for cross-lingual second-stage reranking."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        use_fp16: bool = True,
        cand_docs_limit: int = 200
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.cand_docs_limit = cand_docs_limit
        self._reranker = None

    def _ensure_loaded(self) -> None:
        if self._reranker is None:
            logger.info(f"Loading Cross-Encoder model {self.model_name} (use_fp16={self.use_fp16})...")
            from FlagEmbedding import FlagReranker
            self._reranker = FlagReranker(self.model_name, use_fp16=self.use_fp16)

    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        query_zh: Optional[str] = None,
        top_docs: int = 100,
        top_chunks: int = 98,
        batch_size: int = 64
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        self._ensure_loaded()

        # 1. Deduplicate by doc_id to gather top candidates for Stage 2
        cand_docs_dict: Dict[int, Tuple[str, float]] = {}  # doc_id -> (chunk_text, stage1_score)
        for cand in candidates:
            if not cand.chunk_text or not cand.chunk_text.strip():
                continue
            did_int = int(cand.doc_id)
            if did_int not in cand_docs_dict:
                cand_docs_dict[did_int] = (cand.chunk_text, cand.score)
                if len(cand_docs_dict) >= self.cand_docs_limit:
                    break

        cand_items = list(cand_docs_dict.items())
        if not cand_items:
            return [], []

        # 2. Form cross-lingual (query, chunk_text) pairs
        rerank_pairs: List[List[str]] = []
        for did_int, (chunk_text, _) in cand_items:
            is_zh = any('\u4e00' <= char <= '\u9fff' for char in chunk_text[:200])
            q_str = query_zh if (is_zh and query_zh and query_zh.strip()) else query
            rerank_pairs.append([q_str, chunk_text])

        # 3. Compute cross-encoder relevance scores
        r_scores = self._reranker.compute_score(
            rerank_pairs,
            batch_size=batch_size,
            max_length=512
        )
        if isinstance(r_scores, (int, float)):
            r_scores = [float(r_scores)]

        # 4. Rank candidates by cross-encoder score
        reranked_cand: List[Tuple[float, int, str]] = []
        for (did_int, (chunk_text, _)), r_score in zip(cand_items, r_scores):
            reranked_cand.append((float(r_score), did_int, chunk_text))

        reranked_cand.sort(key=lambda x: x[0], reverse=True)

        # 5. Extract top documents and top unique chunks
        relevant_docs = [int(did) for _, did, _ in reranked_cand[:top_docs]]
        seen_chunk_texts = set()
        relevant_chunks: List[Dict[str, Any]] = []

        for _, did, ctxt in reranked_cand:
            if ctxt not in seen_chunk_texts and len(relevant_chunks) < top_chunks:
                seen_chunk_texts.add(ctxt)
                relevant_chunks.append({
                    "doc_id": int(did),
                    "chunk_text": ctxt
                })

        return relevant_docs, relevant_chunks
