"""Records shared between pipeline stages.

Qrels (dev labels) deliberately use the same shape as a submission entry, so the local
evaluator compares two lists of `QueryResult`. Storage-level schemas for documents and
chunks are documented in docs/design/architecture.md (section "Data contracts").
"""

from __future__ import annotations

from dataclasses import dataclass, field

DocId = str  # ids are compared as strings internally; the writer casts to the dashboard's type


@dataclass(frozen=True)
class ChunkRef:
    doc_id: DocId
    chunk_text: str


@dataclass
class QueryResult:
    id: int
    relevant_docs: list[DocId] = field(default_factory=list)
    relevant_chunks: list[ChunkRef] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> QueryResult:
        return cls(
            id=int(d["id"]),
            relevant_docs=[str(x) for x in d.get("relevant_docs", [])],
            relevant_chunks=[
                ChunkRef(doc_id=str(c["doc_id"]), chunk_text=c["chunk_text"])
                for c in d.get("relevant_chunks", [])
            ],
        )
