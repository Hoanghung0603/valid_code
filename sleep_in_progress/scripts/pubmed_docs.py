"""Xuất doc text cho embed từ pubmed_text.jsonl.zst: {id, idx, text}, text = title + "\\n" + abstract.

Bỏ bài rỗng cả title lẫn abstract. idx = số thứ tự liên tiếp 0..n-1 qua các part (dùng cho cache).
Ghi nhiều part *.jsonl.zst trong --outdir để chuyển SFTP (mỗi part ~ --part-size bài, < 100MiB).
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def doc_text(title: str, abstract: str) -> str:
    """Văn bản doc = title + "\\n" + abstract (nguyên văn từng phần)."""
    return f"{title}\n{abstract}"


def is_empty_doc(title: str, abstract: str) -> bool:
    return not (title or "").strip() and not (abstract or "").strip()


def iter_texts(paths: list) -> dict:
    """Duyệt pubmed_text.jsonl(.zst) -> từng {id, pmid, title, abstract}."""
    import zstandard
    for p in paths:
        raw = open(p, "rb")
        if str(p).endswith(".zst"):
            reader = zstandard.ZstdDecompressor().stream_reader(raw)
        else:
            reader = raw
        buf = b""
        try:
            while True:
                chunk = reader.read(1 << 20)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for ln in lines:
                    if ln:
                        yield json.loads(ln)
            if buf.strip():
                yield json.loads(buf)
        finally:
            reader.close()
            raw.close()


def export_docs(in_paths: list, outdir: Path, part_size: int = 40000) -> dict:
    """Ghi docs_pm.partNNN.jsonl.zst {id, idx, text}; trả thống kê."""
    import zstandard
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    n_in = n_out = n_skip = 0
    pi = 0
    writer = None
    raw = None

    def open_part(i):
        nonlocal raw, writer
        f = outdir / f"docs_pm.part{i:03d}.jsonl.zst"
        raw = f.open("wb")
        writer = zstandard.ZstdCompressor(level=10).stream_writer(raw)

    def close_part():
        nonlocal raw, writer
        if writer is not None:
            writer.close()
            raw.close()
            writer = raw = None

    open_part(pi)
    in_part = 0
    for r in iter_texts(in_paths):
        n_in += 1
        title, abstract = r.get("title") or "", r.get("abstract") or ""
        if is_empty_doc(title, abstract):
            n_skip += 1
            continue
        if in_part >= part_size:
            close_part()
            pi += 1
            open_part(pi)
            in_part = 0
        rec = {"id": int(r["id"]), "idx": n_out, "text": doc_text(title, abstract)}
        writer.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        n_out += 1
        in_part += 1
        if n_out % 50000 == 0:
            print(f"{n_out} doc ({time.time() - t0:.0f}s)", flush=True)
    close_part()
    stats = {"n_in": n_in, "n_docs": n_out, "skipped_empty": n_skip,
             "n_parts": pi + 1, "elapsed": round(time.time() - t0, 1)}
    (outdir / "export_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"export: {n_in} bài -> {n_out} doc ({n_skip} rỗng cả hai), {pi + 1} part, {stats['elapsed']}s",
          flush=True)
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texts", required=True, nargs="+", help="pubmed_text.jsonl.zst (một hoặc nhiều file)")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--part-size", type=int, default=40000)
    a = ap.parse_args()
    export_docs([Path(p) for p in a.texts], Path(a.outdir), a.part_size)


if __name__ == "__main__":
    main()
