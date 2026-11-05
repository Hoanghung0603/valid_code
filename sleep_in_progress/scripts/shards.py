"""Ghi/đọc shard JSONL nén zstd, chịu được file bị cắt ngang khi tiến trình chết.

Mỗi FLUSH_EVERY bản ghi, writer đóng một zstd frame -> crash chỉ mất tối đa FLUSH_EVERY bản ghi cuối.
"""
import io
import json
from pathlib import Path
from typing import Iterator

import zstandard as zstd

FLUSH_EVERY = 200


class ShardWriter:
    def __init__(self, out_dir: Path, prefix: str, max_records: int = 20000, level: int = 6):
        self.out_dir, self.prefix, self.max_records, self.level = out_dir, prefix, max_records, level
        out_dir.mkdir(parents=True, exist_ok=True)
        self._idx = len(list(out_dir.glob(f"{prefix}-*.jsonl.zst")))
        self._fh = self._zw = None
        self._n = 0

    def _open(self):
        path = self.out_dir / f"{self.prefix}-{self._idx:05d}.jsonl.zst"
        self._idx += 1
        self._fh = path.open("wb")
        self._zw = zstd.ZstdCompressor(level=self.level).stream_writer(self._fh)
        self._n = 0

    def write(self, rec: dict):
        if self._zw is None:
            self._open()
        self._zw.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        self._n += 1
        if self._n % FLUSH_EVERY == 0:
            self._zw.flush(zstd.FLUSH_FRAME)
            self._fh.flush()
        if self._n >= self.max_records:
            self.close()

    def close(self):
        if self._zw is not None:
            self._zw.flush(zstd.FLUSH_FRAME)
            self._zw.close()  # đóng luôn file handle
            self._zw = self._fh = None


def read_shard(path: Path) -> Iterator[dict]:
    """Đọc mọi bản ghi hoàn chỉnh; bỏ qua phần đuôi hỏng."""
    data = path.read_bytes()
    dctx = zstd.ZstdDecompressor()
    buf = io.BytesIO()
    try:
        with dctx.stream_reader(io.BytesIO(data), read_across_frames=True) as r:
            while chunk := r.read(1 << 20):
                buf.write(chunk)
    except zstd.ZstdError:
        pass  # frame cuối bị cắt
    for line in buf.getvalue().decode("utf-8", errors="ignore").splitlines():
        try:
            yield json.loads(line)
        except json.JSONDecodeError:
            continue  # dòng cuối dở dang


def iter_records(out_dir: Path, prefix: str = "crawl") -> Iterator[dict]:
    for p in sorted(out_dir.glob(f"{prefix}-*.jsonl.zst")):
        yield from read_shard(p)
