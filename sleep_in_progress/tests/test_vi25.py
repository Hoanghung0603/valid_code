import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vi25


def record(d, text):
    return {"id": d, "status": 200, "text": text, "host": "test.vn", "title": "title"}


def test_prep_drops_vi24_overlap_and_outside_vi_set():
    text = "noi dung y khoa " * 30
    from boilerplate import _norm as norm_line
    kept, bp, rep = vi25.prep_records(
        [record(1, text), record(2, text), record(3, text), record(4, "ngan")],
        {1}, {"test.vn": [norm_line(text)]}, vi_ids={2, 4})
    assert [r["id"] for r in kept] == [2]
    assert rep["overlap_vi24"] == 1
    assert rep["ngoai_tap_vi"] == 1
    assert rep["new"] == 1
    assert rep["empty_after_boilerplate"] == 1
    assert bp["test.vn"] == [norm_line(text)]


def test_prep_splits_outside_ids_by_corpus_membership():
    text = "noi dung y khoa " * 30
    kept, _bp, rep = vi25.prep_records(
        [record(1, text), record(2, text)], set(), {}, vi_ids=set(), all_ids={1})
    assert kept == []
    assert rep["ngoai_tap_vi"] == 2
    assert rep["ngoai_kho"] == 1
    assert rep["trong_kho_khong_phai_vi"] == 1


def test_chain_args_use_vi24_base_and_vi25_outputs():
    args = list(map(str, vi25.chain_args(vi25.B, Path("cands"), "km", "khong-moi")))
    assert args[args.index("--pool") + 1] == str(vi25.B / "cache_pool_v24.json")
    assert args[args.index("--dense") + 1] == str(vi25.B / "dense_v24_top200.json")
    assert args[args.index("--hyde") + 1] == str(vi25.B / "hyde_v24_top200.json")
    assert args[args.index("--ref-kq") + 1] == str(vi25.REF_K)
    assert args[args.index("--out-dir") + 1] == str(vi25.W)
    args = list(map(str, vi25.chain_args(vi25.W, Path("cands25"), "vi25_z93")))
    assert args[args.index("--pool") + 1] == str(vi25.W / "cache_pool_v25.json")
    assert args[args.index("--hyde") + 1] == str(vi25.W / "hyde_v25_top200.json")


def test_sanity_delta_is_vi24_final_delta():
    assert vi25.SANITY_DELTA == 46
    args = list(map(str, vi25.chain_args(vi25.B, Path("cands"), "km", "khong-moi",
                                         delta=vi25.SANITY_DELTA)))
    assert args[args.index("--delta") + 1] == "46"


def test_zip_targets_are_93_and_99():
    assert [n for n, _ in vi25.ZIPS] == ["vi25_z93", "vi25_z99"]
    assert [z for _, z in vi25.ZIPS] == [93_000_000, 99_000_000]


def fake_chain(max_delta_fit):
    """zip tuyen tinh theo delta; code 0 neu <= tran (xay dung boi monkeypatch)."""
    def run(name, delta, max_zip, validate=False):
        zb = 80_000_000 + 250_000 * delta
        fits = delta <= max_delta_fit
        return {"code": 0 if fits else 3,
                "next_delta": None if fits else max(max_delta_fit, 0),
                "zip_bytes": zb, "json_bytes": 1_000_000 * (100 + delta),
                "tong_chunk": 1200 * (100 + delta),
                "hist_kcap": [[100, 300, 1200]]}
    return run


def test_find_delta_picks_largest_delta_with_real_zip_under_ceiling(monkeypatch):
    monkeypatch.setattr(vi25, "chain_zip_check", fake_chain(max_delta_fit=52))
    d, attempts = vi25.find_delta(93_000_000, 48, "probe")
    assert d == 52
    assert attempts[-1]["delta"] >= 52
    assert all(a["code"] == 0 for a in attempts if a["delta"] <= 52)


def test_find_delta_decreases_when_start_is_over_ceiling(monkeypatch):
    monkeypatch.setattr(vi25, "chain_zip_check", fake_chain(max_delta_fit=30))
    d, _attempts = vi25.find_delta(93_000_000, 60, "probe")
    assert d == 30


def test_find_delta_stops_when_zip_saturates(monkeypatch):
    def run(name, delta, max_zip, validate=False):
        return {"code": 0, "next_delta": None, "zip_bytes": 90_000_000,
                "json_bytes": 600_000_000, "tong_chunk": 200000, "hist_kcap": [[100, 120, 1200]]}
    monkeypatch.setattr(vi25, "chain_zip_check", run)
    d, attempts = vi25.find_delta(93_000_000, 5, "probe")
    assert d == 5
    assert len(attempts) <= 4


def test_vinmec_stats_reads_max_latency(tmp_path):
    (tmp_path / "embed.log").write_text("vinmec 200 0.21 ok True\nvinmec 200 1.9 ok False\n",
                                        encoding="utf-8")
    (tmp_path / "rr.log").write_text("vinmec 200 0.35 ok True\n", encoding="utf-8")
    st = vi25.vinmec_stats(tmp_path)
    assert st == {"so_lan": 3, "max_s": 1.9}


def test_shared_helpers_reused_from_vi24():
    import vi24
    assert vi25.owned_lock is vi24.owned_lock
    assert vi25.merge_chunks is vi24.merge_chunks
    assert vi25.gpu_run is vi24.gpu_run
    assert vi25.missing_docwin is vi24.missing_docwin
