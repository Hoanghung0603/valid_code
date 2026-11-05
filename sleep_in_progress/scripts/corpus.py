"""Đọc kho đã crawl (shard *.jsonl.zst) thành danh sách tài liệu sạch, loại trùng id."""
from pathlib import Path
from typing import Iterator

from chunking import detect_lang
from shards import read_shard


def iter_docs(shard_dir: Path, langs: set[str] | None = None, min_chars: int = 200,
              limit: int | None = None) -> Iterator[dict]:
    """Sinh {'id','url','host','title','date','text','lang'} cho bản ghi status 200 có text đủ dài.

    Trùng id (do resume/ghi lại) -> giữ bản đầu tiên. langs: lọc theo {'vi','zh'}.
    """
    seen: set[int] = set()
    n = 0
    for path in sorted(Path(shard_dir).glob("*.jsonl.zst")):
        for r in read_shard(path):
            doc_id = r.get("id")
            text = r.get("text") or ""
            if doc_id is None or doc_id in seen or r.get("status") != 200 or len(text) < min_chars:
                continue
            lang = detect_lang(text)
            if langs and lang not in langs:
                continue
            seen.add(doc_id)
            yield {"id": doc_id, "url": r.get("url", ""), "host": r.get("host", ""), "title": r.get("title", ""),
                   "date": r.get("date", ""), "text": text, "lang": lang}
            n += 1
            if limit and n >= limit:
                return
