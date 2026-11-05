"""Structure-aware chunking: pack document blocks into retrieval units sized for the embedding model.

Sizes are counted with the embedding / reranker tokenizer (default BGE-M3, shared by bge-m3 and
bge-reranker-v2-m3) because the model input limit is what bounds a unit; the official chunk metric no
longer prescribes a tokenizer. Units are (start, end) character spans of `documents.text`, so
`text[start:end]` is always verbatim.
  1. A document that fits in `max_tokens` is one unit.
  2. Otherwise blocks are grouped into sections (a heading starts a section). With `section_atoms`
     (pack-v2) a section that fits `max_tokens` is never split, and consecutive sections are packed
     greedily up to `target_tokens`; larger sections are packed block by block. A unit never ends on a
     heading.
  3. A block longer than `max_tokens` is split at sentence ends (then clause marks, then token windows).
  4. A unit shorter than `min_tokens` is merged into its neighbour when the result fits `max_tokens`.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass

CountFn = Callable[[list[str]], list[int]]  # batched token counter

_SENT_END = re.compile(r"(?<=[。！？!?；;…])(?![”’」』）)\"'])|(?<=[.!?])\s+(?=\S)|\n")
_CLAUSE_END = re.compile(r"(?<=[，,、：:])")


@dataclass
class ChunkConfig:
    target_tokens: int = 256
    max_tokens: int = 512
    min_tokens: int = 64
    section_atoms: bool = True  # pack-v2: keep every section that fits max_tokens in one unit
    version: str = "pack-v2.1"


@dataclass
class Unit:
    start: int
    end: int
    n_tokens: int
    heading_path: str  # headings enclosing the unit's first block (its context at the start)
    is_heading: bool = False


def hf_counter(name: str = "BAAI/bge-m3") -> CountFn:
    from medret.eval.tokenize import _hf_tokenizer

    tok = _hf_tokenizer(name)

    def count(texts: list[str]) -> list[int]:
        return [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)] if texts else []

    return count


def _split_points(text: str, pattern: re.Pattern) -> list[int]:
    cuts = sorted({m.end() for m in pattern.finditer(text)} - {0, len(text)})
    return [0, *cuts, len(text)]


def _split_long(text: str, base: int, n_tokens: int, cfg: ChunkConfig, count: CountFn) -> list[tuple[int, int, int]]:
    """Split one oversize block into pieces <= max_tokens, preferring sentence then clause boundaries."""
    for pattern in (_SENT_END, _CLAUSE_END):
        pts = _split_points(text, pattern)
        pieces = [(a, b) for a, b in zip(pts, pts[1:], strict=False) if text[a:b].strip()]
        sizes = count([text[a:b] for a, b in pieces])
        if len(pieces) > 1 and max(sizes) <= cfg.max_tokens:
            return [(base + a, base + b, n) for (a, b), n in _pack_spans(pieces, sizes, cfg.target_tokens)]
    # no usable boundary: fixed windows by characters, sized from the average chars per token
    step = max(1, int(len(text) * cfg.target_tokens / max(n_tokens, 1)))
    spans = [(a, min(a + step, len(text))) for a in range(0, len(text), step)]
    return [(base + a, base + b, n) for (a, b), n in zip(spans, count([text[a:b] for a, b in spans]), strict=True)]


def _pack_spans(spans: Sequence[tuple[int, int]], sizes: Sequence[int], target: int):
    out, cur, n = [], None, 0
    for (a, b), s in zip(spans, sizes, strict=True):
        if cur is not None and n + s > target:
            out.append((cur, n))
            cur, n = None, 0
        cur = (a, b) if cur is None else (cur[0], b)
        n += s
    if cur is not None:
        out.append((cur, n))
    return out


def chunk_document(text: str, blocks: list[dict], count: CountFn, cfg: ChunkConfig) -> list[Unit]:
    if not blocks:
        return []
    sizes = count([text[b["start"] : b["end"]] for b in blocks])
    total = sum(sizes)
    if total <= cfg.max_tokens:
        return [Unit(blocks[0]["start"], blocks[-1]["end"], total, "")]

    # atoms: blocks, with oversize blocks pre-split; each atom remembers its block's heading flag/path
    atoms: list[Unit] = []
    for b, n in zip(blocks, sizes, strict=True):
        is_h = b["kind"].startswith("h")
        if n > cfg.max_tokens:
            for s, e, m in _split_long(text[b["start"] : b["end"]], b["start"], n, cfg, count):
                atoms.append(Unit(s, e, m, b["heading_path"]))
        else:
            atoms.append(Unit(b["start"], b["end"], n, b["heading_path"], is_h))
    if cfg.section_atoms:
        atoms = _section_atoms(atoms, cfg.max_tokens)

    units: list[Unit] = []
    cur: Unit | None = None
    for a in atoms:
        # pack-v1: a heading starts a new unit unless the current one is still tiny.
        # pack-v2: whole sections are atoms, so consecutive small sections share a unit up to the target.
        starts_section = (
            not cfg.section_atoms and a.is_heading and cur is not None and cur.n_tokens >= cfg.min_tokens
        )
        overflow = cur is not None and cur.n_tokens + a.n_tokens > cfg.target_tokens
        # a unit should not end on a heading, unless keeping it open would pass max_tokens: some pages use
        # runs of <h6> as list items (filmographies), which would otherwise grow one unit without bound
        hard_cap = cur is not None and cur.n_tokens + a.n_tokens > cfg.max_tokens
        if cur is not None and (starts_section or overflow) and (not cur.is_heading or hard_cap):
            units.append(cur)
            cur = None
        if cur is None:
            cur = Unit(a.start, a.end, a.n_tokens, a.heading_path, a.is_heading)
        else:
            cur = Unit(cur.start, a.end, cur.n_tokens + a.n_tokens, cur.heading_path, a.is_heading)
    if cur is not None:
        units.append(cur)

    # merge tiny units into a neighbour (previous first) when it fits
    merged: list[Unit] = []
    for u in units:
        if merged and (u.n_tokens < cfg.min_tokens or merged[-1].n_tokens < cfg.min_tokens) and (
            merged[-1].n_tokens + u.n_tokens <= cfg.max_tokens
        ):
            p = merged[-1]
            merged[-1] = Unit(p.start, u.end, p.n_tokens + u.n_tokens, p.heading_path)
        else:
            merged.append(u)
    return merged


def _section_atoms(atoms: list[Unit], max_tokens: int) -> list[Unit]:
    """Collapse each section (a heading atom and the atoms up to the next heading) that fits in max_tokens
    into one atom, so packing never splits a section that could stay whole."""
    sections: list[list[Unit]] = []
    for a in atoms:
        if a.is_heading or not sections:
            sections.append([a])
        else:
            sections[-1].append(a)
    out: list[Unit] = []
    for sec in sections:
        total = sum(a.n_tokens for a in sec)
        if len(sec) > 1 and total <= max_tokens:
            out.append(Unit(sec[0].start, sec[-1].end, total, sec[0].heading_path))
        else:
            out.extend(sec)
    return out
