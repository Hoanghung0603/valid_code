import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vi23  # noqa: E402
import vi29  # noqa: E402
import vi31  # noqa: E402
import vi33  # noqa: E402
import vi35  # noqa: E402


def _toy():
    # 5 doc; bge xep a>b>c>d>e, 8B xep c>b>a, dense xep b>a>c, hyde xep a>b>c
    pool = {"1": [["a", 0, 1.0], ["b", 1, 0.5], ["c", 2, 0.1], ["d", 3, 0.05], ["e", 4, 0.01]]}
    q8b = {"1": [["c", 2, 3.0], ["b", 1, 2.0], ["a", 0, 1.0]]}
    dense = {"1": [["b", 1, 0.9], ["a", 0, 0.8], ["c", 2, 0.7]]}
    hyde = {"1": [["a", 0, 0.9], ["b", 1, 0.8], ["c", 2, 0.7]]}
    return pool, q8b, dense, hyde


def _fhyde(pool, q8b, dense, hyde):
    from fuse3 import fuse3
    return fuse3(pool, q8b, dense, (vi23.W_BGE, vi23.W_Q8B, vi23.W_DENSE, vi23.W_HYDE),
                 vi23.K_RRF, hyde, None)


def _run(nqf, lo=1, hi=5, thr=-2.0):
    pool, q8b, dense, hyde = _toy()
    old = vi23.NQ_FUSED
    vi23.NQ_FUSED = nqf
    try:
        return vi29.chain_build_v29(pool, q8b, dense, hyde, {}, {}, {}, lo=lo, hi=hi, thr=thr,
                                    dedup_mode="doc_text")
    finally:
        vi23.NQ_FUSED = old


def test_import_khong_dong_vi31_vi33():
    assert vi31.OUT.name == "vi31" and vi33.OUT.name == "vi33"
    assert vi33.DOCLIST_EXTRA == [] and vi23.NQ_FUSED is None


def test_nq_fused_none_giong_truoc(monkeypatch):
    import diag13

    def _phai_khong_duoc_goi(*a, **k):
        raise AssertionError("pick_nq_fused khong duoc goi khi NQ_FUSED None")

    monkeypatch.setattr(diag13, "pick_nq_fused", _phai_khong_duoc_goi)
    from union_sub import adaptive_n
    pool, q8b, dense, hyde = _toy()
    rrf, fh, sub, rep = _run(None)
    assert fh == _fhyde(pool, q8b, dense, hyde)          # fhyde khong doi
    assert rep["nq_mode"] == "bge" and rep["nq_fused_thr"] is None
    ns = [adaptive_n(pool[q], -2.0, 1, 5) for q in sorted(pool, key=int)]
    assert [len(s["relevant_docs"]) for s in sub] == ns  # ref rong -> do dai = n_q
    assert ns == [5]                                     # moi logit > -2.0 -> 5 doc
    # y hett truoc: chay lai khi thuoc tinh NQ_FUSED khong ton tai van giong
    old = vi23.NQ_FUSED
    del vi23.NQ_FUSED
    try:
        rrf2, fh2, sub2, rep2 = vi29.chain_build_v29(
            pool, q8b, dense, hyde, {}, {}, {}, lo=1, hi=5, thr=-2.0, dedup_mode="doc_text")
    finally:
        vi23.NQ_FUSED = old
    assert (rrf2, fh2, sub2) == (rrf, fh, sub)
    assert rep2["nq_mode"] == "bge"


def test_nq_fused_ratio_chon_n_theo_fhyde():
    from diag13 import pick_nq_fused
    pool, q8b, dense, hyde = _toy()
    fh = _fhyde(pool, q8b, dense, hyde)
    t = 0.99
    _rrf, fh2, sub, rep = _run(("ratio", t))
    assert fh2 == fh
    assert rep["nq_mode"] == "fused_ratio" and rep["nq_fused_thr"] == t
    for s in sub:
        q = str(s["id"])
        n = pick_nq_fused(fh[q], t, 1, 5, "ratio")
        assert len(s["relevant_docs"]) == n == rep["n_q_min"] == rep["n_q_max"]
    # ratio 0.99: chi 2 doc dau (ty le diem > 0.99) -> khac adaptive_n (5)
    assert rep["n_q_TB"] == 2.0
    assert [str(d) for d in sub[0]["relevant_docs"]] == [str(h[0]) for h in fh["1"][:2]]


def test_chain_voi_nqf_phai_trung_pick_nq_fused_moi_cau():
    from diag13 import pick_nq_fused
    pool, q8b, dense, hyde = _toy()
    pool["2"] = [["x", 0, 0.9], ["y", 1, 0.4], ["z", 2, -3.0]]
    q8b["2"] = [["y", 1, 5.0], ["x", 0, 1.0]]
    dense["2"] = [["x", 0, 0.5]]
    hyde["2"] = [["y", 1, 0.5]]
    fh = _fhyde(pool, q8b, dense, hyde)
    old = vi23.NQ_FUSED
    vi23.NQ_FUSED = ("ratio", 0.5)
    try:
        _rrf, _fh, sub, rep = vi29.chain_build_v29(pool, q8b, dense, hyde, {}, {}, {},
                                                   lo=1, hi=4, thr=-2.0, dedup_mode="doc_text")
    finally:
        vi23.NQ_FUSED = old
    for s in sub:
        q = str(s["id"])
        n = pick_nq_fused(fh[q], 0.5, 1, 4, "ratio")
        assert len(s["relevant_docs"]) == n
    assert rep["n_q_min"] == min(pick_nq_fused(fh[q], 0.5, 1, 4, "ratio") for q in fh)
    assert rep["n_q_max"] == max(pick_nq_fused(fh[q], 0.5, 1, 4, "ratio") for q in fh)


def test_cmd_doclist_fused_khop_chain_khong_co_thi_nhu_cu(tmp_path):
    from diag13 import pick_nq_fused
    from union_sub import adaptive_n
    pool, q8b, dense, hyde = _toy()
    fh = _fhyde(pool, q8b, dense, hyde)
    p_pool, p_fh = tmp_path / "pool.json", tmp_path / "fhyde.json"
    p_pool.write_text(json.dumps(pool), encoding="utf-8")
    p_fh.write_text(json.dumps(fh), encoding="utf-8")

    def run_dl(method, fthr, out):
        a = SimpleNamespace(pool=str(p_pool), fhyde=str(p_fh), ref_chunks="", thr=-2.0,
                            lo=1, hi=5, out=str(out), report="",
                            nq_fused_method=method, nq_fused_thr=fthr)
        assert vi23.cmd_doclist(a) == 0
        return json.loads(out.read_text(encoding="utf-8"))

    t = 0.99
    old = vi23.NQ_FUSED
    vi23.NQ_FUSED = ("ratio", t)
    try:
        _rrf, _fh, sub, _rep = vi29.chain_build_v29(pool, q8b, dense, hyde, {}, {}, {},
                                                    lo=1, hi=5, thr=-2.0, dedup_mode="doc_text")
    finally:
        vi23.NQ_FUSED = old
    got = run_dl("ratio", t, tmp_path / "dl_fused.json")
    assert got == {str(s["id"]): [str(d) for d in s["relevant_docs"]] for s in sub}
    assert got["1"] == [str(h[0]) for h in fh["1"][: pick_nq_fused(fh["1"], t, 1, 5, "ratio")]]
    # khong co co -> nhu cu: adaptive_n + first_n_docs fhyde
    got0 = run_dl(None, None, tmp_path / "dl_old.json")
    n0 = adaptive_n(pool["1"], -2.0, 1, 5)
    assert n0 == 5
    assert got0["1"] == [str(h[0]) for h in fh["1"][:n0]]
    assert got0["1"] != got["1"]


def test_cmd_doclist_method_thieu_thr_bao_loi(tmp_path):
    pool, q8b, dense, hyde = _toy()
    fh = _fhyde(pool, q8b, dense, hyde)
    p_pool, p_fh = tmp_path / "pool.json", tmp_path / "fhyde.json"
    p_pool.write_text(json.dumps(pool), encoding="utf-8")
    p_fh.write_text(json.dumps(fh), encoding="utf-8")
    a = SimpleNamespace(pool=str(p_pool), fhyde=str(p_fh), ref_chunks="", thr=-2.0,
                        lo=1, hi=5, out=str(tmp_path / "dl.json"), report="",
                        nq_fused_method="ratio", nq_fused_thr=None)
    try:
        vi23.cmd_doclist(a)
    except SystemExit as e:
        assert "nq-fused-thr" in str(e)
    else:
        raise AssertionError("phai bao loi khi thieu --nq-fused-thr")


def test_bind_unbind_vi35_tra_lai_bien_cu():
    assert vi33.DOCLIST_EXTRA == []
    old = vi35.bind()
    try:
        assert vi31.OUT.name == "vi35" and vi31.G.name == "vi35"
        assert vi31.Q8_V31.as_posix().endswith("diag33/q8b_v33.json")
        assert vi31.CHUNK30.parent.name == "vi34" and vi31.CHUNK30.name == "chunk_dw_u900_v34.json.zst"
        assert vi31.CHUNK31.name == "chunk_dw_u900_v35.json.zst"
        assert vi31.NAME == "vi35" and vi31.DELTA0 == 56
        assert vi33.Q8 == vi35.Q8 and vi33.OUT32.name == "vi34"
        assert vi33.W8F == 2.0 and vi33.DELTA == 56
        assert vi33.DOCLIST_EXTRA == []
    finally:
        vi35.unbind(old)
    assert vi31.OUT.name == "vi31" and vi33.OUT.name == "vi33"
    assert vi33.DOCLIST_EXTRA == [] and vi33.W8F == 2.0


def test_vi35_hang_so_va_stage():
    assert vi35.OUT == vi35.ROOT / "data/sub/vi35" and vi35.G == vi35.ROOT / "work/vi35"
    assert (vi35.NAME, vi35.SANITY_NAME, vi35.PROBE) == ("vi35", "vi35_sanity", "vi35p")
    assert vi35.DELTA == 56 and vi35.W8F == 2.0 and vi35.NQ_METHOD == "ratio"
    assert vi35.Q8.as_posix().endswith("diag33/q8b_v33.json")
    assert vi35.FHYDE34 == vi35.OUT34 / "cache_fhyde_v31.json"
    assert set(vi35.STAGES) == {"sanity", "calib", "doclist", "chunk", "build", "stats", "docwin"}
