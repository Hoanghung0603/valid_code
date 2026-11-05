"""Evidence-linked graph fallback for every official query.

This is a visualization of cached retrieval associations, not a clinical KG
or a new submission ranker. Edges are either normalized lexical overlaps with
submitted chunks or explicit cached query->document links.
"""
from __future__ import annotations

import re
import unicodedata

STOP = {"anh", "bac", "ban", "benh", "biet", "cho", "co", "con", "cua", "da", "dang",
        "de", "duoc", "em", "gi", "hay", "hoi", "khi", "khong", "lam", "la", "lieu",
        "mot", "nao", "nay", "nguoi", "nhu", "nhung", "phai", "sao", "suc", "the",
        "thi", "toi", "trong", "va", "voi", "xin", "can", "nen", "theo", "roi",
        "bai", "bacsi", "theo", "sau", "truoc", "cac", "hay", "the", "nao",
        "the", "dieu", "tri", "thuoc", "dau", "cam", "thay", "lam", "sao"}


def fold(text: str) -> str:
    value = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in value if not unicodedata.combining(char))


def query_anchors(query: str, limit: int = 6) -> list[str]:
    """Salient literal tokens; no LLM diagnosis or inferred synonym edges."""
    folded = fold(query)
    words = re.findall(r"[a-z0-9]{3,}|[\u3400-\u9fff]{2,}", folded)
    unique = list(dict.fromkeys(word for word in words if word not in STOP))
    # Prefer medical-looking longer words while preserving query order for ties.
    return sorted(unique, key=lambda word: -min(len(word), 12))[:limit]


def build_replay_graph(query_id: int, query: str, rows: list[dict], limit: int = 8) -> dict:
    anchors = query_anchors(query)
    candidates = []
    for row in rows:
        text = str(row.get("chunk_text") or "")
        folded = fold(text)
        hits = [term for term in anchors if re.search(r"(?<!\w)" + re.escape(term) + r"(?!\w)", folded)]
        candidates.append((row, hits))
    matched = [(row, hits) for row, hits in candidates if hits]
    matched.sort(key=lambda pair: (-len(pair[1]), int(pair[0].get("chunk_rank", 10**9))))
    pool = matched if matched else candidates
    chosen = []
    selected_ids = set()
    for row, hits in pool:
        if row["doc_id"] in selected_ids:
            continue
        chosen.append((row, hits))
        selected_ids.add(row["doc_id"])
        if len(chosen) >= limit:
            break
    connected = list(dict.fromkeys(term for _, hits in chosen for term in hits))
    direct = not connected
    nodes = ([{"id": "replay_path", "label": "cached retrieval", "kind": "replay_path", "chapter": None}]
             if direct else [{"id": f"term:{term}", "label": term, "kind": "query_term", "chapter": None}
                             for term in connected])
    docs = []
    seen = set()
    for row, hits in chosen:
        doc_id = int(row["doc_id"])
        if doc_id in seen:
            continue
        seen.add(doc_id)
        text = str(row.get("chunk_text") or "")
        evidence = ([{"concept_id": f"term:{term}", "alias": term, "source": "chunk_overlap",
                      "chunk_ref": row.get("chunk_ref"), "snippet": text[:180]} for term in hits]
                    if not direct else [{"concept_id": "replay_path", "alias": "cached prediction",
                                           "source": "replay_rank", "chunk_ref": row.get("chunk_ref"),
                                           "snippet": text[:180]}])
        docs.append({"doc_id": doc_id, "url": row.get("url"),
                     "overlap": len(hits), "rank": row.get("doc_rank"),
                     "chunk_rank": row.get("chunk_rank"), "evidence": evidence})
    return {"query_id": query_id, "query": query, "mode": "replay_evidence",
            "scored": False, "affects_submission": False, "concepts": nodes,
            "documents": docs, "edge_basis": "replay_rank" if direct else "normalized_chunk_overlap",
            "warning": "Cached Exp143 retrieval and lexical overlap only; not a medical relation or relevance label."}
