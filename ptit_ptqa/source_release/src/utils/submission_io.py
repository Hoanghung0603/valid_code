"""Streaming and provenance helpers for competition submission files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator


def sha256_file(path: str | Path, block_size: int = 4 * 1024 * 1024) -> str:
    """Return a SHA-256 digest without loading the whole file into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: str | Path, data: Any) -> None:
    """Write deterministic, human-readable UTF-8 JSON."""
    Path(path).write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def iter_json_array(path: str | Path, block_size: int = 256 * 1024) -> Iterator[Any]:
    """Yield members of a top-level JSON array with bounded transient memory."""
    decoder = json.JSONDecoder()
    with Path(path).open("r", encoding="utf-8") as handle:
        buffer = ""
        began = False
        eof = False
        while True:
            if not buffer and not eof:
                block = handle.read(block_size)
                eof = not block
                buffer += block

            buffer = buffer.lstrip()
            if not began:
                if not buffer or buffer[0] != "[":
                    raise ValueError("Expected a top-level JSON array")
                buffer = buffer[1:]
                began = True
                continue

            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                return

            try:
                item, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise
                block = handle.read(block_size)
                eof = not block
                buffer += block
                continue

            yield item
            buffer = buffer[end:]
