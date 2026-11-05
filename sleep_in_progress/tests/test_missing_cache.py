import json
import sys
from pathlib import Path

import zstandard

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from missing_cache import doc_ids, missing_cache  # noqa: E402


def test_missing_cache_topn_only_missing():
    cache = {"1": [[1, 10, 0.9], [2, 20, 0.8], [3, 30, 0.7]], "2": [[2, 20, 0.5], [4, 40, 0.4]]}
    out = missing_cache(cache, {"1", "4"}, n=2)
    assert out == {"1": [[2, 20, 0.8]], "2": [[2, 20, 0.5]]}


def test_doc_ids_reads_parts(tmp_path):
    p = tmp_path / "d.jsonl.zst"
    raw = "".join(json.dumps({"id": i, "idx": i, "text": "x"}) + "\n" for i in (5, 6))
    p.write_bytes(zstandard.ZstdCompressor().compress(raw.encode()))
    assert doc_ids([str(p)]) == {"5", "6"}
