"""Write, validate and zip submission files (docs/about-competition/submission-instruction.md)."""

from __future__ import annotations

import io
import itertools
import json
import zipfile
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from medret.schemas import QueryResult


def _zipped(path: Path) -> tuple[Path, str] | None:
    """The zip next to `path` that holds a member named path.name (runs keep only the zip to save disk)."""
    for zp in sorted(path.parent.glob("*.zip")) if path.parent.is_dir() else []:
        with zipfile.ZipFile(zp) as zf:
            if path.name in zf.namelist():
                return zp, path.name
    return None


def submission_exists(path: str | Path) -> bool:
    path = Path(path)
    return path.exists() or _zipped(path) is not None


def _read_text(path: Path) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    if (hit := _zipped(path)) is not None:
        with zipfile.ZipFile(hit[0]) as zf:
            return zf.read(hit[1]).decode("utf-8")
    raise FileNotFoundError(path)


def read_submission(path: str | Path) -> list[dict]:
    """Raw entries of a submission or qrels file (.jsonl also accepted). When the JSON was deleted after packing,
    it is read from the run's zip (any zip in the same folder with a member of the same name)."""
    path = Path(path)
    text = _read_text(path)
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


@contextmanager
def _open_text(path: Path) -> Iterator[io.TextIOBase]:
    if path.exists():
        with open(path, encoding="utf-8") as f:
            yield f
    elif (hit := _zipped(path)) is not None:
        with zipfile.ZipFile(hit[0]) as zf, io.TextIOWrapper(zf.open(hit[1]), encoding="utf-8") as f:
            yield f
    else:
        raise FileNotFoundError(path)


def iter_submission(path: str | Path, block_chars: int = 1 << 24) -> Iterator[dict]:
    """read_submission one entry at a time, reading the file in blocks: neither the whole text (~2 GB as a str for
    a 1 GB JSON) nor all entries as objects (~5 GB) is held, which exhausted the dev laptop on 2026-10-26."""
    path = Path(path)
    with _open_text(path) as f:
        if path.suffix == ".jsonl":
            yield from (json.loads(line) for line in f if line.strip())
            return
        dec, buf, i, eof, started = json.JSONDecoder(), "", 0, False, False
        while True:
            while i < len(buf) and buf[i] in " \t\r\n," + ("" if started else "["):
                started = started or buf[i] == "["
                i += 1
            if i < len(buf) and buf[i] == "]":
                return
            try:
                if i >= len(buf):
                    raise ValueError("need more data")
                item, i = dec.raw_decode(buf, i)
                yield item
                continue
            except ValueError:  # json.JSONDecodeError included: the entry runs past the block
                if eof:
                    if i >= len(buf):
                        return
                    raise
            block = f.read(block_chars)
            eof = not block
            buf, i = buf[i:] + block, 0


def load_results(path: str | Path) -> list[QueryResult]:
    """Reads a submission or a qrels file (same JSON shape; .jsonl also accepted), see read_submission."""
    return [QueryResult.from_dict(d) for d in read_submission(path)]


def _cast(doc_id: str, doc_id_type: Literal["int", "str"]) -> int | str:
    return int(doc_id) if doc_id_type == "int" else str(doc_id)


def write_submission(
    results: Iterable[QueryResult], path: str | Path, doc_id_type: Literal["int", "str"] = "int"
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # streamed one query at a time: whole-document submissions reach ~1 GB of JSON, and one json.dumps of the
    # payload held it twice in RAM next to the texts (the dev laptop ran out of memory, 2026-10-19)
    with open(path, "w", encoding="utf-8") as f:
        f.write("[")
        for i, r in enumerate(sorted(results, key=lambda r: r.id)):
            item = {
                "id": int(r.id),
                "relevant_docs": [_cast(d, doc_id_type) for d in r.relevant_docs],
                "relevant_chunks": [
                    {"doc_id": _cast(c.doc_id, doc_id_type), "chunk_text": c.chunk_text}
                    for c in r.relevant_chunks
                ],
            }
            f.write((", " if i else "") + json.dumps(item, ensure_ascii=False))
        f.write("]")
    return path


def validate_submission(
    path: str | Path,
    query_ids: Sequence[int] | None = None,
    corpus_ids: set[str] | None = None,
) -> list[str]:
    """Returns a list of problems; empty means the file matches the spec. Streams the entries (iter_submission), so
    a 1 GB file is checked without holding it as objects."""
    errors: list[str] = []
    path = Path(path)
    try:
        with open(path, encoding="utf-8") as f:
            head = f.read(4096).lstrip()
    except OSError as e:
        return [f"cannot read JSON: {e}"]
    if not head.startswith("["):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            return [f"cannot read JSON: {e}"]
        return ["top level must be a JSON array"]

    seen: set[int] = set()
    entries = iter_submission(path)
    for i in itertools.count():
        try:
            item = next(entries)
        except StopIteration:
            break
        except ValueError as e:  # json.JSONDecodeError included
            errors.append(f"cannot read JSON: {e}")
            break
        where = f"entry #{i}"
        if not isinstance(item, dict):
            errors.append(f"{where}: not an object")
            continue
        qid = item.get("id")
        if not isinstance(qid, int) or isinstance(qid, bool):
            errors.append(f"{where}: id must be an integer, got {qid!r}")
        elif qid in seen:
            errors.append(f"{where}: duplicate id {qid}")
        else:
            seen.add(qid)
            where = f"query {qid}"

        docs = item.get("relevant_docs")
        if not isinstance(docs, list):
            errors.append(f"{where}: relevant_docs must be a list (use [] when empty)")
            docs = []
        chunks = item.get("relevant_chunks")
        if not isinstance(chunks, list):
            errors.append(f"{where}: relevant_chunks must be a list (use [] when empty)")
            chunks = []

        for d in docs:
            if corpus_ids is not None and str(d) not in corpus_ids:
                errors.append(f"{where}: doc {d!r} not in links_corpus")
        for j, c in enumerate(chunks):
            if not isinstance(c, dict) or "doc_id" not in c or not isinstance(c.get("chunk_text"), str):
                errors.append(f"{where}: chunk #{j} needs doc_id and string chunk_text")
                continue
            if corpus_ids is not None and str(c["doc_id"]) not in corpus_ids:
                errors.append(f"{where}: chunk #{j} doc {c['doc_id']!r} not in links_corpus")
            order = c.get("chunk_order")
            if order is not None and (not isinstance(order, int) or isinstance(order, bool) or order < 0):
                errors.append(f"{where}: chunk #{j} chunk_order must be a non-negative integer")

    if query_ids is not None:
        missing = set(query_ids) - seen
        extra = seen - set(query_ids)
        if missing:
            errors.append(f"{len(missing)} query ids missing (e.g. {sorted(missing)[:5]})")
        if extra:
            errors.append(f"{len(extra)} unknown query ids (e.g. {sorted(extra)[:5]})")
    return errors


def overlap_order(items: list[dict]) -> list[int]:
    """Query order for compression: a greedy chain where each next query shares the most chunk docs with the last.

    Whole-document chunks of one article repeat across the queries it answers; adjacent repeats fall inside the
    LZMA dictionary and cost almost nothing (52.8 MB instead of 79.2 MB on the 0.3076 submission, 2026-10-14).
    The JSON array order of the queries carries no meaning for the scorer (entries are matched by id).
    """
    import numpy as np
    from scipy.sparse import csr_matrix

    sets = [{str(c["doc_id"]) for c in it.get("relevant_chunks", [])} for it in items]
    col = {d: i for i, d in enumerate(sorted({d for s in sets for d in s}))}
    rows = [i for i, s in enumerate(sets) for _ in s]
    cols = [col[d] for s in sets for d in s]
    m = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(sets), max(len(col), 1)))
    ov = (m @ m.T).toarray()
    np.fill_diagonal(ov, -1)
    order, seen = [0], np.zeros(len(sets), dtype=bool)
    seen[0] = True
    for _ in range(len(sets) - 1):
        nxt = int(np.argmax(np.where(seen, -2, ov[order[-1]])))
        order.append(nxt)
        seen[nxt] = True
    return order


class _BigDictLZMACompressor:
    """zipfile LZMA compressor with a chosen LZMA1 dictionary size. The size travels in the entry's LZMA
    properties, which zipfile's decompressor (Python >= 3.3, the grader runs 3.7) reads back."""

    dict_mb = 256
    match_finder = "bt4"  # "hc4": ~1.9 GB instead of ~2.9 GB of encoder memory at 256 MB; decoding is unaffected

    def __init__(self):
        self._comp = None

    def _init(self):
        import lzma
        import struct

        props = lzma._encode_filter_properties({"id": lzma.FILTER_LZMA1, "preset": 6,
                                                "dict_size": self.dict_mb << 20})
        filt = lzma._decode_filter_properties(lzma.FILTER_LZMA1, props)
        if self.match_finder == "hc4":  # encoder-only setting, not part of the stored properties
            filt["mf"] = lzma.MF_HC4
        self._comp = lzma.LZMACompressor(lzma.FORMAT_RAW, filters=[filt])
        return struct.pack("<BBH", 9, 4, len(props)) + props

    def compress(self, data):
        return (self._init() if self._comp is None else b"") + self._comp.compress(data)

    def flush(self):
        return (self._init() if self._comp is None else b"") + self._comp.flush()


def reorder_submission(json_path: str | Path) -> Path:
    """Rewrite the JSON in place with the queries in overlap_order (smaller LZMA zip; order carries no meaning).

    Streamed, freeing each entry once written: a 750 MB JSON held as objects plus one dumped string plus the ~3 GB
    LZMA encoder (256 MB dictionary) exhausted the 14 GB dev laptop (2026-10-14). For ~1 GB JSONs, run this and the
    zip step in separate processes: CPython keeps the freed object memory, so the encoder came on top of it.
    Since 2026-10-26 the entries are streamed (iter_submission) into a side file of serialised entries and copied
    back by byte offset, so only the chunk doc ids are held: loading a 1 GB JSON as objects no longer fit.
    """
    json_path = Path(json_path)
    side = json_path.with_suffix(".entries.tmp")
    light, spans = [], []
    with open(side, "wb") as tmp:
        for item in iter_submission(json_path):
            light.append({"relevant_chunks": [{"doc_id": c["doc_id"]} for c in item.get("relevant_chunks", [])]})
            data = json.dumps(item, ensure_ascii=False).encode("utf-8")
            spans.append((tmp.tell(), len(data)))
            tmp.write(data)
    order = overlap_order(light)
    del light
    with open(side, "rb") as tmp, open(json_path, "wb") as f:
        f.write(b"[")
        for j, i in enumerate(order):
            tmp.seek(spans[i][0])
            f.write((b", " if j else b"") + tmp.read(spans[i][1]))
        f.write(b"]")
    side.unlink()
    return json_path


def pack_submission(json_path: str | Path, zip_path: str | Path | None = None, method: str = "lzma",
                    dict_mb: int | None = None, reorder: bool = False, match_finder: str = "bt4") -> Path:
    """ZIP containing exactly one JSON file at the archive root, as the dashboard requires.

    LZMA by default: the leaderboard reads LZMA zips (probe test-lzma-deep-full40-v1 scored exactly like its
    deflate twin, 2026-10-14), and whole-document chunks repeat across queries, which LZMA's large window removes:
    52.9 MB instead of 85.1 MB with deflate. Upload limit: 100 MB (organisers, 2026-10-19); uploads are slow.
    dict_mb: LZMA1 dictionary in MB (zipfile's default is 8); reorder: write the queries in overlap_order so
    repeated documents sit close together. Both shrink whole-document submissions by about a third.
    """
    json_path = Path(json_path)
    zip_path = Path(zip_path) if zip_path else json_path.with_suffix(".zip")
    if reorder:
        reorder_submission(json_path)
    kw = {"compression": zipfile.ZIP_LZMA} if method == "lzma" else {"compression": zipfile.ZIP_DEFLATED,
                                                                      "compresslevel": 9}
    patched = method == "lzma" and dict_mb is not None
    if patched:
        _BigDictLZMACompressor.dict_mb = dict_mb
        _BigDictLZMACompressor.match_finder = match_finder
        default, zipfile.LZMACompressor = zipfile.LZMACompressor, _BigDictLZMACompressor
    try:
        with zipfile.ZipFile(zip_path, "w", **kw) as zf:
            zf.write(json_path, arcname=json_path.name)
    finally:
        if patched:
            zipfile.LZMACompressor = default
    return zip_path
