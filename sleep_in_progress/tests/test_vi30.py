import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vi30


def test_constants():
    assert vi30.MAX_ZIP == 99_000_000
    assert vi30.DELTA0 == 58 and vi30.SANITY_DELTA == 58
    assert vi30.EXP_ADD == 26769 and vi30.N_PKG == 122859 and vi30.N_BAD == 16079
    assert vi30.OUT == vi30.ROOT / "data/sub/diag30"
    assert vi30.IN29 == vi30.ROOT / "data/sub/diag29/in1"
    assert len(vi30.PKG_FILES) == 7


def test_select_add_ids_ba_ca():
    idx = {1, 2, 3, 4}          # chi muc cu
    bad = {3, 4, 9}             # 3,4 nhiem trong chi muc; 9 nhiem khong crawl lai
    pkg = {1, 3, 5}             # 1 cu sach, 3 bad crawl lai, 5 moi
    sel = vi30.select_add_ids(pkg, idx, bad)
    assert sel["skip"] == {1}               # id cu khong bad giu nguyen
    assert sel["recraw"] == {3}             # id bad crawl lai -> them
    assert sel["fresh"] == {5}              # id moi -> them
    assert sel["add"] == {3, 5}


def test_giu_id_cu_khong_co_trong_goi():
    # id 2 (cu, sach) khong co trong goi -> van giu trong chi muc moi
    rep = vi30.verify_index_ids([1, 2, 3, 5], [1, 2, 3, 4], {3, 4}, [3, 5])
    assert rep["ok"] and rep["mong_doi"] == 4


def test_filter_rows_bo_bad():
    rows = [{"id": 1}, {"id": 3}, {"id": 4, "x": 1}, {"title": "khong id"}]
    out = vi30.filter_rows(rows, {3, 4})
    assert [r.get("id") for r in out] == [1, None]


def test_verify_index_ids_bat_loi():
    assert not vi30.verify_index_ids([1, 1, 2], [1, 2], set(), [])["ok"]          # trung
    assert not vi30.verify_index_ids([1, 2, 4], [1, 2, 4], {4}, [])["ok"]         # con id bad cu
    assert not vi30.verify_index_ids([1], [1, 2], set(), [])["ok"]                # thieu
    r = vi30.verify_index_ids([1, 2, 4], [1, 2, 4], {4}, [4])                    # bad crawl lai
    assert r["ok"]


def test_merge_dense_bo_hit_cu_bad_idempotent():
    old = {"1": [[3, 0, 0.9], [1, 1, 0.5]]}     # 3 la embedding text cu
    new = {"1": [[3, 7, 0.4], [5, 8, 0.6]]}     # 3 nhung lai tren text moi
    out, rep = vi30.merge_dense_old_new(old, new, {3}, 3)
    assert out["1"] == [[5, 8, 0.6], [1, 1, 0.5], [3, 7, 0.4]]
    assert rep["hit_cu_bad_bo"] == 1
    out2, _ = vi30.merge_dense_old_new(vi30.drop_stale_hits(old, {3}), new, {3}, 3)
    assert out2 == out
    assert vi30.drop_stale_hits(vi30.drop_stale_hits(old, {3}), {3}) == vi30.drop_stale_hits(old, {3})


def test_split_docwin_khong_dung_lai_p3_cua_doc_them():
    cands = {"1": [[10, 0], [3, 1], [5, 2]]}
    p3 = {"1": {"10": [1.0], "3": [2.0]}}
    reuse, fresh = vi30.split_docwin_cands(cands, p3, {"3", "5"})
    assert reuse == {"1": [[10, 0]]}
    assert fresh == {"1": [[3, 1], [5, 2]]}


def test_build_excl():
    assert vi30.build_excl({3, 4, 9}, [3, 5]) == {4, 9}


def test_assert_clean_dung_tap_loai_vi30():
    import vi29
    sub = [{"id": i, "relevant_docs": [3, 5] if i == 1 else [],
            "relevant_chunks": [{"doc_id": 3, "chunk_text": "a"}] if i == 1 else []}
           for i in range(1, 1201)]
    excl = vi30.build_excl({3, 4}, [3])
    vi29.assert_clean(sub, excl, {3, 5})          # 3 crawl lai hop le
    with pytest.raises(AssertionError):
        vi29.assert_clean(sub, {3, 4}, {3, 5})     # neu loai ca bad thi phai loi


def test_check_wincheck():
    vi30.check_wincheck({"so_cap": 300, "so_mau": 300, "max_sai": 0.01})
    with pytest.raises(SystemExit):
        vi30.check_wincheck({"so_cap": 299, "so_mau": 300, "max_sai": 0.01})
    with pytest.raises(SystemExit):
        vi30.check_wincheck({"so_cap": 300, "so_mau": 300, "max_sai": 0.03})


def test_fp4_status():
    sel = vi30.select_add_ids({7, 8}, {8}, {7})
    st = vi30.fp4_status([7, 8], sel, {7}, {8}, {7}, {7, 8})
    assert st["7"]["giu_sau_loc"] and st["7"]["con_trong_chi_muc_vi30"]
    assert not st["8"]["duoc_chon"] and st["8"]["con_trong_chi_muc_vi30"]


def test_read_ids_tsv(tmp_path):
    p = tmp_path / "b.tsv"
    p.write_text("id\treason\n5\tx\n7\ty\n\n", encoding="utf-8")
    assert vi30.read_ids_tsv(p) == [5, 7]


def test_write_zst_lines(tmp_path):
    import vi23
    p = tmp_path / "a.jsonl.zst"
    assert vi30.write_zst_lines(p, [{"id": 1}, {"id": 2}]) == 2
    assert [r["id"] for r in vi23.read_shard_records(p)] == [1, 2]
