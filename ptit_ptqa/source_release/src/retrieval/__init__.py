"""Retrieval module for multilingual dense and sparse search."""

from src.retrieval.base import (
    BaseFusionStrategy,
    BaseQueryEncoder,
    Candidate,
    QueryBundle,
    RetrievalResult,
)
from src.retrieval.query_encoder import BGEM3QueryEncoder
from src.retrieval.index_repository import IndexRepository, IndexShard
from src.retrieval.fusion import LinearFusionStrategy, RRFFusionStrategy
from src.retrieval.searcher import VectorSearcher

__all__ = [
    "BaseFusionStrategy",
    "BaseQueryEncoder",
    "Candidate",
    "QueryBundle",
    "RetrievalResult",
    "BGEM3QueryEncoder",
    "IndexRepository",
    "IndexShard",
    "LinearFusionStrategy",
    "RRFFusionStrategy",
    "VectorSearcher",
]
