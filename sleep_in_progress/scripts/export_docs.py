"""Xuất text của các doc trong chỉ mục VI để đem đi embed (bi-encoder) trên máy GPU.

Hai chế độ:
  - mặc định: lấy doc duy nhất trong top-N của cache BM25 (search_cache.py), ghi JSONL nén zstd
    {"id","idx","text"} (tiêu đề + text đã bỏ boilerplate, cắt --chars ký tự) + file ứng viên.
  - --all: xuất MỌI doc của chỉ mục (cùng cách làm sạch/cắt text như trên), chia nhiều part
    --part-mb (nén, ước lượng thích nghi) để chuyển SFTP ổn định. Không cần cache/ứng viên.
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import zstandard

from bigindex import BigIndex
from bigpipeline import load_texts
from boilerplate import BoilerplateFilter
from shards import read_shard


def make_body(title: str, text: str, chars: int, no_title: bool = False) -> str:
    if no_title:
        return text[:chars]
    return (title + "\n" + text).strip()[:chars]


def iter_all_texts(idx, wanted: dict[int, int], bp: BoilerplateFilter, chars: int, log=print,
                   no_title: bool = False):
    """Yield (doc_id, idx, body) cho mọi doc trong wanted {doc_id: idx}, gom theo shard.

    Text giống hệt load_texts (bản ghi status 200 ĐẦU TIÊN của id trong shard, cùng làm sạch).
    """
    by_shard: dict[str, set[int]] = defaultdict(set)
    for doc_id, i in wanted.items():
        by_shard[idx.shard_of(i)].add(doc_id)
    for n, (shard, ids) in enumerate(sorted(by_shard.items()), 1):
        left = set(ids)
        out = []
        for r in read_shard(Path(shard)):
            did = r.get("id")
            if did in left and r.get("status") == 200:
                left.discard(did)
                host = r.get("host", "")
                title = r.get("title") or ""
                out.append((did, wanted[did], make_body(title, bp.clean(host, r.get("text") or ""), chars, no_title)))
                if not left:
                    break
        if left:
            log(f"CẢNH BÁO: {len(left)} doc không đọc lại được từ {shard}")
        if n % 10 == 0 or n == len(by_shard):
            log(f"đọc shard {n}/{len(by_shard)}")
        yield from out


class PartWriter:
    """Ghi JSONL.zst; xoay part khi vượt --part-mb (nén, ước lượng từ tỉ lệ nén của part trước)."""

    def __init__(self, out: str, part_mb: int = 0, log=print):
        self.out = Path(out)
        self.part_mb = part_mb
        self.log = log
        self.limit = part_mb * 1024 * 1024
        self.raw_budget = 3 * self.limit if part_mb > 0 else 0
        self.paths: list[Path] = []
        self.pi = 0
        self.n = 0
        self._raw = 0
        self._fh = self._zw = None

    def _path(self) -> Path:
        if self.part_mb <= 0:
            return self.out
        name = self.out.name
        stem = name[: -len(".jsonl.zst")] if name.endswith(".jsonl.zst") else name
        return self.out.parent / f"{stem}.part{self.pi:03d}.jsonl.zst"

    def _open(self) -> None:
        p = self._path()
        self._fh = p.open("wb")
        self._zw = zstandard.ZstdCompressor(level=6).stream_writer(self._fh)
        self.paths.append(p)
        self._raw = 0

    def write(self, obj: dict) -> None:
        if self._zw is None:
            self._open()
        line = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
        self._zw.write(line)
        self._raw += len(line)
        self.n += 1
        if self.part_mb > 0 and self._raw >= self.raw_budget:
            self._close_part()

    def _close_part(self) -> None:
        if self._zw is None:
            return
        self._zw.close()
        self._zw = self._fh = None
        sz = self.paths[-1].stat().st_size
        if sz > 0:
            self.raw_budget = min(4 * self.limit, max(1 << 20, int(self.limit * (self._raw / sz) * 0.95)))
        self.log(f"part {self.paths[-1].name}: {sz / 1e6:.0f}MB")
        self.pi += 1

    def close(self) -> None:
        self._close_part()
        if self.part_mb <= 0 and self.paths:
            self.log(f"ghi {self.paths[-1].name}: {self.paths[-1].stat().st_size / 1e6:.0f}MB")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--cache", help="JSON cache search_cache.py (bỏ qua khi --all)")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--chars", type=int, default=3000)
    ap.add_argument("--out-docs", required=True, help="JSONL.zst (khi --part-mb>0: tiền tố partNNN)")
    ap.add_argument("--out-cands", help="JSON ứng viên (bắt buộc khi không --all)")
    ap.add_argument("--all", action="store_true", help="xuất mọi doc của chỉ mục")
    ap.add_argument("--part-mb", type=int, default=0, help="xoay part khi file nén vượt N MB (ước lượng)")
    ap.add_argument("--no-title", action="store_true", help="body = text sạch[:chars], KHÔNG prepend title")
    a = ap.parse_args()

    t0 = time.time()
    idx = BigIndex(Path(a.index))
    bp = BoilerplateFilter.from_json((Path(a.index) / "boilerplate.json").read_text(encoding="utf-8"))

    if a.all:
        wanted = {}
        for i in range(idx.n):
            wanted[idx.doc_id(i)] = i
        print(f"{idx.n} doc trong chỉ mục, {len(wanted)} id duy nhất ({time.time() - t0:.0f}s)", flush=True)
        w = PartWriter(a.out_docs, a.part_mb, log=lambda m: print(m, flush=True))
        for did, i, body in iter_all_texts(idx, wanted, bp, a.chars, log=lambda m: print(m, flush=True),
                                          no_title=a.no_title):
            w.write({"id": did, "idx": i, "text": body})
            if w.n % 50000 == 0:
                print(f"{w.n} doc ({time.time() - t0:.0f}s)", flush=True)
        w.close()
        print(f"ghi {w.n} doc vào {len(w.paths)} part, thiếu {len(wanted) - w.n} doc; tổng {time.time() - t0:.0f}s",
              flush=True)
        return

    if not a.cache or not a.out_cands:
        ap.error("cần --cache và --out-cands khi không --all")
    cache = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    cands = {q: [[int(d), int(i)] for d, i, _ in hits[: a.n]] for q, hits in cache.items()}
    wanted = {d: i for hits in cands.values() for d, i in hits}
    print(f"{len(cands)} câu, {len(wanted)} doc duy nhất trong top-{a.n} ({time.time() - t0:.0f}s)", flush=True)

    texts = load_texts(idx, wanted, bp, log=lambda m: print(m, flush=True))

    missing = [d for d in wanted if d not in texts]
    w = PartWriter(a.out_docs, a.part_mb, log=lambda m: print(m, flush=True))
    for d, i in wanted.items():
        r = texts.get(d)
        if not r:
            continue
        w.write({"id": d, "idx": i, "text": make_body(r["title"], r["text"], a.chars, a.no_title)})
    w.close()
    n = w.n
    Path(a.out_cands).write_text(json.dumps(cands), encoding="utf-8")
    size = sum(p.stat().st_size for p in w.paths) / 1e6
    print(f"ghi {n} doc vào {a.out_docs} ({size:.0f}MB, {len(w.paths)} part), thiếu {len(missing)} doc; "
          f"ứng viên -> {a.out_cands}; tổng {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
