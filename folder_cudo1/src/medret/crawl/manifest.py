"""Crawl manifest: one row per fetched doc id, pointing at its raw HTML inside the page shards.

Page shards are `crawl_dir/pages/<site>/part-*.parquet` (zstd), one row per fetch attempt with columns
id, url, final_url, status, label, detail, ctype, ms, fetched_at, html. This is the layout of the team
crawler (packs under crawl_dir/_packs/); `html` holds the raw response bytes and is only set when
label == "ok". Archive copies carry `detail = "archive:wb:<timestamp> ..."` or `"archive:cc:<crawl> ..."`.

Later crawl snapshots (same row schema, plus `site`, `lang`) live in `crawl_dir/pages/snap_<date>/html_*.parquet`.

Manifest columns follow docs/design/architecture.md section 3: `shard` is the shard path relative to
crawl_dir and `offset` is the row index inside that shard. When an id was fetched more than once, a successful
fetch wins over failures; among successful fetches the earliest pack wins (pack_rank: the original pages first,
then snapshots by date), so documents that are already parsed, chunked and embedded never change source;
within one pack the latest successful fetch wins. Without any success, the latest attempt is kept.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PAGES_SUBDIR = "pages"
MANIFEST_FILE = "manifest.parquet"
ALIASES_FILE = "aliases.parquet"
_META_COLUMNS = ["id", "url", "final_url", "status", "label", "detail", "ctype", "fetched_at"]


SNAPSHOT_PREFIX = "snap_"


def page_shards(crawl_dir: Path) -> list[Path]:
    """Original crawler shards first, then every snapshot directory in date order."""
    pages = crawl_dir / PAGES_SUBDIR
    snaps = sorted(d for d in pages.glob(f"{SNAPSHOT_PREFIX}*") if d.is_dir())
    return sorted(pages.glob("*/part-*.parquet")) + [f for d in snaps for f in sorted(d.glob("html_*.parquet"))]


def pack_rank(shard: Path, crawl_dir: Path) -> int:
    """0 for the original crawl, i for the i-th snapshot (in date order)."""
    rel = shard.relative_to(crawl_dir / PAGES_SUBDIR).parts[0]
    if not rel.startswith(SNAPSHOT_PREFIX):
        return 0
    snaps = sorted(d.name for d in (crawl_dir / PAGES_SUBDIR).glob(f"{SNAPSHOT_PREFIX}*") if d.is_dir())
    return snaps.index(rel) + 1


def _source(detail: str) -> str:
    if detail.startswith("archive:wb:"):
        return "wayback"
    if detail.startswith("archive:cc:"):
        return "commoncrawl"
    return "live"


def scan_shard(path: Path, crawl_dir: Path) -> pd.DataFrame:
    """Manifest rows (before dedup) for every fetch attempt in one shard, hashing the HTML batch by batch."""
    pf = pq.ParquetFile(path)
    meta = pf.read(columns=_META_COLUMNS).to_pandas()
    lengths, hashes = [], []
    for batch in pf.iter_batches(columns=["html"], batch_size=512):
        for raw in batch.column(0).to_pylist():
            lengths.append(len(raw) if raw else 0)
            hashes.append(hashlib.sha1(raw).hexdigest() if raw else "")
    detail = meta["detail"].fillna("")
    label = meta["label"].fillna("")
    return pd.DataFrame(
        {
            "doc_id": meta["id"].astype("int64"),
            "url": meta["url"],
            "final_url": meta["final_url"],
            "status": meta["status"].fillna(-1).astype("int32"),
            "content_type": meta["ctype"].fillna(""),
            "source": detail.map(_source),
            "fetched_at": pd.to_datetime(meta["fetched_at"], unit="s", utc=True),
            "shard": path.relative_to(crawl_dir).as_posix(),
            "offset": range(len(meta)),
            "length": lengths,
            "sha1": hashes,
            "error": label.where(label != "ok", ""),
            "detail": detail,
            "pack_rank": pack_rank(path, crawl_dir),
        }
    ).astype({"offset": "int64", "length": "int64"})


def build_manifest(crawl_dir: Path, workers: int | None = None) -> pd.DataFrame:
    shards = page_shards(crawl_dir)
    if not shards:
        raise FileNotFoundError(f"no page shards under {crawl_dir / PAGES_SUBDIR}")
    with ProcessPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(scan_shard, shards, [crawl_dir] * len(shards), chunksize=8))
    rows = pd.concat(parts, ignore_index=True)
    # keep="last" after sorting: success over failure, then the earliest pack, then the latest fetch
    rows["_ok"] = rows["error"].eq("") & rows["length"].gt(0)
    rows["_rank"] = -rows["pack_rank"]
    rows = rows.sort_values(["doc_id", "_ok", "_rank", "fetched_at"], kind="stable")
    return rows.drop_duplicates("doc_id", keep="last").drop(columns=["_ok", "_rank"]).reset_index(drop=True)


def build_aliases(corpus: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Map every corpus id to a fetched OK id with the same `canonical_key` (exact URL duplicates only).

    Columns: id, key (canonical_key), fetched_id (Int64, <NA> when no URL of the group was fetched OK).
    """
    from medret.data.urls import canonical_key

    out = corpus[["id"]].copy()
    out["key"] = corpus["url"].map(canonical_key)
    ok_ids = manifest.loc[manifest["error"].eq(""), "doc_id"]
    fetched = out[out["id"].isin(ok_ids)].drop_duplicates("key")
    out = out.merge(fetched.rename(columns={"id": "fetched_id"}), on="key", how="left")
    out["fetched_id"] = out["fetched_id"].astype("Int64")
    return out


def read_html_rows(shard_file: Path | str, offsets) -> Iterator[tuple[int, bytes]]:
    """Yield (offset, raw html) for the given row offsets of one shard, one row group at a time.

    Snapshot shards reach 10.8 GB of HTML (pages/snap_20261012/html_zh-0004.parquet, 333k rows in 500-row
    groups); reading the whole `html` column of such a shard exhausted the laptop's RAM twice (2026-10-14).
    Only the row groups that hold a wanted offset are read, so memory stays at about one group (~20 MB).
    """
    f = pq.ParquetFile(shard_file)
    starts, pos = [], 0
    for i in range(f.metadata.num_row_groups):
        starts.append(pos)
        pos += f.metadata.row_group(i).num_rows
    import bisect

    want: dict[int, list[int]] = {}
    for off in sorted(int(o) for o in offsets):
        want.setdefault(bisect.bisect_right(starts, off) - 1, []).append(off)
    for g, offs in want.items():
        col = f.read_row_group(g, columns=["html"]).column("html")
        for off in offs:
            yield off, col[off - starts[g]].as_py()
        del col


def iter_pages(crawl_dir: Path, manifest: pd.DataFrame) -> Iterator[tuple[int, bytes, str]]:
    """Yield (doc_id, raw html, content_type) for the OK manifest rows given, shard by shard (row-group reads)."""
    ok = manifest[manifest["error"].eq("") & manifest["length"].gt(0)]
    for shard, group in ok.groupby("shard", sort=True):
        by_off = {int(o): (int(d), c) for o, d, c in zip(group["offset"], group["doc_id"], group["content_type"],
                                                          strict=True)}
        for off, raw in read_html_rows(crawl_dir / shard, by_off):
            doc_id, ctype = by_off[off]
            yield doc_id, raw, ctype
