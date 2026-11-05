"""Make remaining_urls.parquet the exact official corpus complement.

Expected partition:
    official = embedded UNION unembedded UNION remaining

All three sets are compared as exact (id, url) pairs and must be disjoint.
The original remaining file is backed up before atomic replacement.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OFFICIAL = DATA_DIR / "raw" / "btc_provided" / "links_corpus_vi_en_zh.parquet"
EMBEDDED = DATA_DIR / "embedded_samples.parquet"
UNEMBEDDED = DATA_DIR / "unembedded_samples.parquet"
REMAINING = DATA_DIR / "remaining_urls.parquet"
BACKUP = DATA_DIR / "remaining_urls.parquet.pre_reconcile_20261016.bak"
TEMP = DATA_DIR / "remaining_urls.reconcile_tmp.parquet"

EXPECTED_BEFORE = 14_652
EXPECTED_ADDED = 184
EXPECTED_AFTER = 14_836


def load_pairs(path: Path) -> tuple[list[int], list[str], set[tuple[int, str]]]:
    table = pq.read_table(path, columns=["id", "url"])
    ids = table["id"].to_pylist()
    urls = table["url"].to_pylist()
    return ids, urls, set(zip(ids, urls))


def main() -> None:
    if BACKUP.exists() or TEMP.exists():
        raise FileExistsError("Backup or temporary output already exists")

    official_ids, official_urls, official_pairs = load_pairs(OFFICIAL)
    _, _, embedded_pairs = load_pairs(EMBEDDED)
    _, _, unembedded_pairs = load_pairs(UNEMBEDDED)
    remaining_ids, remaining_urls, current_pairs = load_pairs(REMAINING)

    if len(current_pairs) != EXPECTED_BEFORE:
        raise RuntimeError(f"Unexpected current remaining size: {len(current_pairs):,}")
    if embedded_pairs & unembedded_pairs:
        raise RuntimeError("embedded and unembedded registries overlap")
    if not embedded_pairs.issubset(official_pairs):
        raise RuntimeError("embedded contains non-official pairs")
    if not unembedded_pairs.issubset(official_pairs):
        raise RuntimeError("unembedded contains non-official pairs")

    covered = embedded_pairs | unembedded_pairs
    expected_pairs = official_pairs - covered
    missing = expected_pairs - current_pairs
    extra = current_pairs - expected_pairs
    if len(missing) != EXPECTED_ADDED or extra:
        raise RuntimeError(
            f"Unexpected reconciliation delta: missing={len(missing):,}, extra={len(extra):,}"
        )

    # Preserve the authoritative corpus order for deterministic crawl batches.
    output_ids: list[int] = []
    output_urls: list[str] = []
    for doc_id, url in zip(official_ids, official_urls):
        if (doc_id, url) in expected_pairs:
            output_ids.append(doc_id)
            output_urls.append(url)

    if len(output_ids) != EXPECTED_AFTER:
        raise RuntimeError(f"Unexpected reconciled size: {len(output_ids):,}")
    if len(set(output_ids)) != EXPECTED_AFTER or len(set(output_urls)) != EXPECTED_AFTER:
        raise RuntimeError("Duplicate IDs or URLs in reconciled remaining set")

    shutil.copy2(REMAINING, BACKUP)
    table = pa.table(
        {
            "id": pa.array(output_ids, type=pa.int64()),
            "url": pa.array(output_urls, type=pa.string()),
        }
    )
    pq.write_table(table, TEMP, compression="snappy")

    # Validate the exact three-way partition from the serialized file.
    check_ids, check_urls, check_pairs = load_pairs(TEMP)
    if check_pairs != expected_pairs:
        raise RuntimeError("Serialized remaining set differs from expected complement")
    if check_pairs & embedded_pairs or check_pairs & unembedded_pairs:
        raise RuntimeError("Reconciled remaining set overlaps another registry")
    if embedded_pairs | unembedded_pairs | check_pairs != official_pairs:
        raise RuntimeError("Three-way partition does not cover the official corpus")
    if len(check_ids) != len(set(check_ids)) or len(check_urls) != len(set(check_urls)):
        raise RuntimeError("Duplicates detected after serialization")

    os.replace(TEMP, REMAINING)
    print(f"remaining URLs: {EXPECTED_BEFORE:,} -> {len(check_pairs):,}")
    print(f"added missing official URLs: {len(missing):,}")
    print("removed extra URLs: 0")
    print(f"recoverable backup: {BACKUP}")


if __name__ == "__main__":
    main()
