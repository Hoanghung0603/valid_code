from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Iterator, TextIO


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_json_array(stream: TextIO, chunk_size: int = 1024 * 1024) -> Iterator[dict]:
    """Incrementally decode a top-level JSON array without loading it into RAM."""
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    started = False
    finished = False
    while not finished:
        block = stream.read(chunk_size)
        if block:
            buffer = buffer[position:] + block
            position = 0
        else:
            buffer = buffer[position:]
            position = 0
            finished = True
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if not started:
                if position >= len(buffer):
                    break
                if buffer[position] != "[":
                    raise ValueError("Submission payload must be a JSON array")
                started = True
                position += 1
                continue
            while position < len(buffer) and (buffer[position].isspace() or buffer[position] == ","):
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            if position >= len(buffer):
                break
            try:
                value, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if finished:
                    raise
                break
            if not isinstance(value, dict):
                raise ValueError("Submission rows must be objects")
            yield value
            position = end
    raise ValueError("Submission JSON array ended unexpectedly")


def iter_submission_rows(path: Path) -> Iterator[dict]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise ValueError("Submission ZIP must contain only a valid submission.json")
        with archive.open("submission.json") as raw:
            with io.TextIOWrapper(raw, encoding="utf-8") as text:
                yield from iter_json_array(text)
