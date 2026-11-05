"""Auditable query -> alias -> concept -> document graph for candidate retrieval.

Only exact corpus mentions become document edges. A concept's ICD chapter is
metadata for inspection, not a diagnosis or an automatic filter.
"""
from __future__ import annotations

import json
import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text)


@dataclass(frozen=True)
class Mention:
    concept_id: str
    alias: str
    start: int  # Offset in NFKC/casefold-normalized text, not raw-source offset.


class ConceptLexicon:
    def __init__(self, path: Path):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "medical_concepts_v1":
            raise ValueError("Unsupported concept lexicon")
        self.concepts = payload["concepts"]
        self.aliases: dict[str, str] = {}
        for concept in self.concepts:
            cid = concept["id"]
            for alias in concept["aliases"]:
                key = normalize(alias).strip()
                min_length = 2 if any("\u3400" <= char <= "\u9fff" for char in key) else 3
                if len(key) < min_length or (key in self.aliases and self.aliases[key] != cid):
                    raise ValueError(f"Invalid or ambiguous alias: {alias}")
                self.aliases[key] = cid
        terms = sorted(self.aliases, key=len, reverse=True)
        self.pattern = re.compile(r"(?<!\w)(?:" + "|".join(map(re.escape, terms)) + r")(?!\w)")

    def mentions(self, text: str, max_chars: int | None = None) -> list[Mention]:
        haystack = normalize(text[:max_chars] if max_chars else text)
        found: dict[str, Mention] = {}
        for match in self.pattern.finditer(haystack):
            alias = match.group()
            cid = self.aliases[alias]
            found.setdefault(cid, Mention(cid, alias, match.start()))
        return list(found.values())


SCHEMA = """
CREATE TABLE concepts (id TEXT PRIMARY KEY, chapter TEXT NOT NULL);
CREATE TABLE aliases (alias TEXT PRIMARY KEY, concept_id TEXT NOT NULL REFERENCES concepts(id));
CREATE TABLE documents (doc_id INTEGER PRIMARY KEY, url TEXT NOT NULL);
CREATE TABLE mentions (doc_id INTEGER NOT NULL REFERENCES documents(doc_id),
                       concept_id TEXT NOT NULL REFERENCES concepts(id),
                       alias TEXT NOT NULL, normalized_offset INTEGER NOT NULL,
                       PRIMARY KEY(doc_id, concept_id));
CREATE INDEX mention_concept ON mentions(concept_id, doc_id);
"""


def create_graph(path: Path, lexicon: ConceptLexicon) -> sqlite3.Connection:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite graph: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO concepts VALUES (?,?)", ((c["id"], c["chapter"]) for c in lexicon.concepts))
    db.executemany("INSERT INTO aliases VALUES (?,?)", lexicon.aliases.items())
    db.commit()
    return db


def add_document(db: sqlite3.Connection, lexicon: ConceptLexicon, doc_id: int, url: str, text: str,
                 max_chars: int = 12000) -> int:
    mentions = lexicon.mentions(text, max_chars=max_chars)
    if not mentions:
        return 0
    db.execute("INSERT INTO documents VALUES (?,?)", (doc_id, url))
    db.executemany("INSERT INTO mentions VALUES (?,?,?,?)",
                   ((doc_id, m.concept_id, m.alias, m.start) for m in mentions))
    return len(mentions)


def retrieve(db: sqlite3.Connection, lexicon: ConceptLexicon, query: str,
             limit: int = 50, max_span: int = 800) -> list[dict]:
    """Return conservative candidates; never infer unseen clinical relationships."""
    concepts = sorted({m.concept_id for m in lexicon.mentions(query)})
    if not concepts or limit < 1:
        return []
    placeholders = ",".join("?" for _ in concepts)
    sql = f"""SELECT m.doc_id, COUNT(*) AS overlap,
                     MIN(m.normalized_offset) AS first_offset,
                     MAX(m.normalized_offset) - MIN(m.normalized_offset) AS span,
                     GROUP_CONCAT(m.concept_id) AS matched_concepts, d.url
              FROM mentions AS m JOIN documents AS d USING(doc_id)
              WHERE m.concept_id IN ({placeholders})
              GROUP BY m.doc_id
              HAVING COUNT(*) = 1 OR MAX(m.normalized_offset) - MIN(m.normalized_offset) <= ?
              ORDER BY overlap DESC, span ASC, first_offset ASC, m.doc_id ASC LIMIT ?"""
    fields = ("doc_id", "overlap", "first_offset", "span", "matched_concepts", "url")
    return [dict(zip(fields, row)) for row in db.execute(sql, (*concepts, max_span, limit))]
