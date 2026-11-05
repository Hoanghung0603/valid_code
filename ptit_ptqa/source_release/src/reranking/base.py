"""Base abstraction for 2-stage rerankers."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.retrieval.base import Candidate


class BaseReranker(ABC):
    """Abstract interface for Stage-2 neural cross-encoder rerankers."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[Candidate],
        query_zh: Optional[str] = None,
        top_docs: int = 100,
        top_chunks: int = 98,
        batch_size: int = 64
    ) -> Tuple[List[int], List[Dict[str, Any]]]:
        """
        Reranks candidates against the query and returns (relevant_docs, relevant_chunks).
        """
        pass
