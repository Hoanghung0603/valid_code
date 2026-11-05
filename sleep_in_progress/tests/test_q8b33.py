import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import q8b31  # noqa: E402
import q8b33  # noqa: E402


def test_q8b33_paths_and_prefix(monkeypatch):
    for k in ("PREFIX", "SELECT_FHYDE", "SELECT_POOL", "SELECT_Q8B", "SELECT_NEW_IDS", "IX", "TOP"):
        monkeypatch.setattr(q8b31, k, getattr(q8b31, k))
    assert q8b33.FHYDE.as_posix() == "D:/r2ai/data/sub/vi33/cache_fhyde_v31.json"
    assert q8b33.OLD.as_posix() == "D:/r2ai/data/sub/vi33/q8b_v31_clean.json"
    assert q8b33.DOCLIST.as_posix() == "D:/r2ai/data/sub/vi33/doclist_vi33.json"
    q8b33.apply_base()
    assert q8b31.PREFIX == "q8b33"
    assert q8b31.SELECT_FHYDE == q8b33.FHYDE and q8b31.SELECT_Q8B == q8b33.OLD


def test_per_query_top_adds_margin():
    assert q8b33.per_query_top({"1": [1, 2, 3], "2": []}, 5) == {"1": 8, "2": 5}


def test_select_pairs_doclist_per_query_depth_skips_scored_and_outside_pool():
    fhyde = {"1": [[d, 0, 1.0] for d in range(1, 11)], "2": [[d, 0, 1.0] for d in range(1, 11)]}
    pool = {"1": [[d, 0, 0] for d in range(1, 11)], "2": [[d, 0, 0] for d in range(1, 10)]}
    q8b = {"1": [[2, 0, 1.0]], "2": []}
    got = q8b33.select_pairs_doclist(fhyde, pool, q8b, {"1": 4, "2": 10})
    assert got == {"1": [1, 3, 4], "2": list(range(1, 10))}


def test_select_pairs_doclist_query_missing_from_tops_gets_none():
    fhyde = {"1": [[1, 0, 1.0]], "9": [[1, 0, 1.0]]}
    pool = {"1": [[1, 0, 0]], "9": [[1, 0, 0]]}
    assert q8b33.select_pairs_doclist(fhyde, pool, {}, {"1": 5}) == {"1": [1], "9": []}


def test_configure_replaces_base_select(monkeypatch):
    monkeypatch.setattr(q8b31, "select_pairs", q8b31.select_pairs)
    q8b33.configure({"1": [1]}, margin=1)
    fhyde = {"1": [[d, 0, 1.0] for d in range(1, 6)]}
    pool = {"1": [[d, 0, 0] for d in range(1, 6)]}
    assert q8b31.select_pairs(fhyde, pool, {}, 170) == {"1": [1, 2]}
