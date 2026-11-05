import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vi26


def test_cap_top_relaxes_by_900_over_u():
    assert vi26.cap_top(900) == 120
    assert vi26.cap_top(600) == 180
    assert vi26.cap_top(450) == 240


def test_layer_specs_keep_vi25_sources_and_only_change_units():
    names = [s["name"] for s in vi26.layer_specs(600)]
    assert names == ["top100", "ext", "v23", "v24", "v25"]
    s900 = {s["name"]: s for s in vi26.layer_specs(900)}
    s600 = {s["name"]: s for s in vi26.layer_specs(600)}
    for name in names:
        assert s900[name]["docs"] == s600[name]["docs"]
        assert s900[name]["wins"] == s600[name]["wins"]
        assert s900[name]["cands"] == s600[name]["cands"]
        assert s900[name]["top_docs"] == s600[name]["top_docs"]
    assert s900["top100"]["top_chunks"] == 120
    assert s600["top100"]["top_chunks"] == 180
    assert s900["ext"]["dq"] and s900["v23"]["dq"]
    assert not s900["top100"]["dq"] and not s900["v25"]["dq"]
    assert s900["v25"]["reuse"].name == "cands_docwin_reuse.json"
    assert s900["v23"]["wins"] == [vi26.P3 / "docwin_p3.json.zst", vi26.V23 / "docwin_vi23.json.zst"]


def test_chain_args_use_vi25_pool_and_diag26_out():
    args = list(map(str, vi26.chain_args(Path("cands"), "vi26_u600", 120)))
    assert args[args.index("--pool") + 1] == str(vi26.W / "cache_pool_v25.json")
    assert args[args.index("--dense") + 1] == str(vi26.W / "dense_v25_top200.json")
    assert args[args.index("--hyde") + 1] == str(vi26.W / "hyde_v25_top200.json")
    assert args[args.index("--q8b") + 1] == str(vi26.Q8)
    assert args[args.index("--cands") + 1] == "cands"
    assert args[args.index("--delta") + 1] == "120"
    assert args[args.index("--out-dir") + 1] == str(vi26.OUT)
    assert args[args.index("--ref-kq") + 1] == str(vi26.REF_K)


def test_zip_specs_prioritise_u600():
    assert vi26.ZIPS == [("vi26_u600", 600), ("vi26_u450", 450)]
    assert vi26.MAX_ZIP == 99_000_000
    assert vi26.SANITY_DELTA == 57


def test_largest_delta_grows_with_budget():
    hist = [[50, 500, 1200]]
    d1 = vi26.largest_delta(hist, 600_000_000, 130_000, 99_000_000, 0.154)
    d2 = vi26.largest_delta(hist, 600_000_000, 130_000, 93_000_000, 0.154)
    assert d1 >= d2
    assert d1 > 0


def _fake(monkeypatch, fits_delta, zip_step=250_000):
    calls = []

    def chain_only(name, u, delta):
        calls.append(("chain", name, delta))
        return {"delta": delta, "json_bytes": 500_000_000, "tong_chunk": 1200 * (50 + delta),
                "hist_kcap": [[50, 1000, 1200]], "TB_chunk_cau": 50 + delta}

    def chain_zip_check(name, u, delta, max_zip):
        calls.append(("zip", name, delta))
        code = 0 if delta <= fits_delta else 3
        return {"code": code, "next_delta": fits_delta if code == 3 else None,
                "zip_bytes": 80_000_000 + zip_step * delta, "json_bytes": 500_000_000,
                "tong_chunk": 1200 * (50 + delta), "hist_kcap": [[50, 1000, 1200]],
                "delta": delta, "name": name}

    monkeypatch.setattr(vi26, "chain_only", chain_only)
    monkeypatch.setattr(vi26, "chain_zip_check", chain_zip_check)
    return calls


def test_find_delta_picks_largest_delta_with_real_zip(monkeypatch):
    calls = _fake(monkeypatch, fits_delta=130)
    d, attempts = vi26.find_delta("vi26_u600", 600, 99_000_000, 120)
    assert d == 130
    assert any(a["delta"] == 130 and a["code"] == 0 and a["final"] for a in attempts)
    assert any(c[0] == "zip" and c[2] == 131 for c in calls)


def test_find_delta_falls_back_to_next_delta_when_start_over(monkeypatch):
    _fake(monkeypatch, fits_delta=100)
    d, attempts = vi26.find_delta("vi26_u600", 600, 99_000_000, 150)
    assert d == 100
    assert any(a["code"] == 3 for a in attempts)


def test_find_delta_without_plus1_stops_at_fitted(monkeypatch):
    calls = _fake(monkeypatch, fits_delta=130)
    d, _attempts = vi26.find_delta("vi26_u600", 600, 99_000_000, 120, plus1=False)
    assert d == 130
    assert not any(c[0] == "zip" and c[2] == 131 for c in calls)


def test_merge_into_counts_duplicate_keys():
    merged = {}
    dup = vi26.merge_into(merged, {"1": [[10, 1, 0, 1.0, "a"]]})
    assert dup == 0
    dup = vi26.merge_into(merged, {"1": [[10, 1, 0, 2.0, "b"]]})
    assert dup == 1
    assert merged["1"][0][4] == "b"


def test_pct_nearest_rank():
    assert vi26.pct([1, 2, 3, 4], 0.5) == 3
    assert vi26.pct([], 0.5) == 0.0
