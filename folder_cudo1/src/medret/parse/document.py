"""Assemble cleaned blocks into a document record (data contract `data/parsed/documents-*.parquet`)."""

from __future__ import annotations

from medret.parse.clean import Block

BLOCK_SEP = "\n"


def assemble(blocks: list[Block]) -> tuple[str, list[dict]]:
    """Join blocks into `text` and return block offsets with their heading path.

    heading_path lists the enclosing headings below the title (h2 > h3 ...), joined by " > ".
    """
    parts: list[str] = []
    spans: list[dict] = []
    stack: list[tuple[int, str]] = []  # (level, text) of open headings, h2..h6
    pos = 0
    for b in blocks:
        if b.kind.startswith("h") and b.kind != "h1":
            level = int(b.kind[1])
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, b.text.replace("\n", " ")))
        if parts:
            pos += len(BLOCK_SEP)
        spans.append(
            {"start": pos, "end": pos + len(b.text), "kind": b.kind, "heading_path": " > ".join(t for _, t in stack)}
        )
        parts.append(b.text)
        pos += len(b.text)
    return BLOCK_SEP.join(parts), spans
