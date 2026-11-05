"""Core abstractions and data transfer objects for retrieval."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set
import numpy as np
import scipy.sparse as sp


@dataclass(frozen=True)
class QueryBundle:
    """Represents a search query, optionally containing multilingual translations."""
    id: int
    text_vi: str
    text_zh: Optional[str] = None
    text_en: Optional[str] = None


@dataclass
class Candidate:
    """Represents a retrieved chunk candidate with metadata."""
    score: float
    doc_id: int
    chunk_text: str
    dense_score: Optional[float] = None
    chunk_id: Optional[int] = None


@dataclass
class RetrievalResult:
    """Standardized output structure for a single query."""
    query_id: int
    relevant_docs: List[int] = field(default_factory=list)
    relevant_chunks: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.query_id),
            "relevant_docs": [int(d) for d in self.relevant_docs],
            "relevant_chunks": self.relevant_chunks,
        }


class BaseQueryEncoder(ABC):
    """Abstract interface for query encoders (dense & sparse)."""

    @abstractmethod
    def encode_dense(
        self,
        queries: List[str],
        batch_size: int = 8,
        max_length: int = 1024
    ) -> np.ndarray:
        """Encodes queries into dense floating-point embedding vectors."""
        pass

    @abstractmethod
    def encode_sparse(
        self,
        queries_vi: List[str],
        queries_zh: Optional[List[str]] = None,
        queries_en: Optional[List[str]] = None,
        zh_weight: float = 1.0,
        en_weight: float = 0.5,
        batch_size: int = 8,
        max_length: int = 1024
    ) -> sp.csr_matrix:
        """Encodes queries into sparse lexical token weight matrix."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Releases underlying model weights and VRAM/RAM resources."""
        pass


class BaseFusionStrategy(ABC):
    """Abstract interface for multi-modal candidate scoring and fusion."""

    @abstractmethod
    def fuse_and_rank(
        self,
        candidates: List[Any],
        dense_weight: float = 1.0,
        sparse_weight: float = 1.0,
        **kwargs: Any
    ) -> List[Candidate]:
        """Combines multimodal scores and returns a ranked list of candidates."""
        pass
