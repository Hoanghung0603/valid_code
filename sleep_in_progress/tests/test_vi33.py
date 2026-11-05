import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import vi23  # noqa: E402
import vi29  # noqa: E402
import vi31  # noqa: E402
import vi33  # noqa: E402


def _toy():
    # 3 doc; bge xep a>b>c, 8B xep c>b>a, dense xep b>a>c, hyde xep a>b>c
    pool = {"1": [["a", 0, 1.0], ["b", 1, 0.5], ["c", 2, 0.1]]}
    q8b = {"1": [["c", 2, 3.0], ["b", 1, 2.0], ["a", 0, 1.0]]}
    dense = {"1": [["b", 1, 0.9], ["a", 0, 0.8], ["c", 2, 0.7]]}
    hyde = {"1": [["a", 0, 0.9], ["b", 1, 0.8], ["c", 2, 0.7]]}
    return pool, q8b, dense, hyde


def test_import_does_not_touch_vi31():
    assert vi31.OUT.name == "vi31" and vi31.NAME == "vi31"


def test_bind_rebinds_then_restore():
    old = vi33.bind()
    try:
        assert vi31.OUT.name == "vi33" and vi31.G.name == "vi33"
        assert vi31.Q8_V31.name == "q8b_v32.json"
        assert vi31.CHUNK30 == vi33.CHUNK_V32 and vi33.CHUNK_V32.parent.name == "vi32"
        assert vi31.CHUNK31.name == "chunk_dw_u900_v33.json.zst"
        assert vi31.NAME == "vi33" and vi31.DELTA0 == 56
    finally:
        for k, v in old.items():
            setattr(vi31, k, v)
    assert vi31.OUT.name == "vi31"


def _chain(w8f):
    pool, q8b, dense, hyde = _toy()
    vi33.set_w8f(w8f)
    try:
        rrf, fh, _, _ = vi29.chain_build_v29(pool, q8b, dense, hyde, {}, {}, {}, lo=1, hi=3,
                                              thr=-99.0, dedup_mode="doc_text")
    finally:
        vi33.set_w8f(None)
    return rrf, fh


def test_default_w8f_is_old_behaviour():
    from fuse3 import fuse3
    pool, q8b, dense, hyde = _toy()
    rrf, fh = _chain(None)
    w = (vi23.W_BGE, vi23.W_Q8B, vi23.W_DENSE, vi23.W_HYDE)
    assert fh == fuse3(pool, q8b, dense, w, vi23.K_RRF, hyde, None)
    assert rrf == fuse3(pool, q8b, dense, w[:3] + (0.0,), vi23.K_RRF, None, None)


def test_w8f_changes_only_fhyde():
    rrf1, fh1 = _chain(None)
    rrf2, fh2 = _chain(2.0)
    assert rrf1 == rrf2                       # zd cho chunk khong doi
    s1 = {h[0]: h[2] for h in fh1["1"]}
    s2 = {h[0]: h[2] for h in fh2["1"]}
    k = vi23.K_RRF
    for d, r8 in {"c": 1, "b": 2, "a": 3}.items():
        assert abs((s2[d] - s1[d]) - 1.0 / (k + r8)) < 1e-6


def test_set_w8f_resets():
    vi33.set_w8f(2.0)
    assert vi23.W_Q8B_FHYDE == 2.0
    vi33.set_w8f(None)
    assert vi23.W_Q8B_FHYDE is None


def test_cmd_fuse_flag(tmp_path):
    import json
    pool, q8b, dense, hyde = _toy()
    paths = {}
    for n, obj in {"pool": pool, "q8b": q8b, "dense": dense, "hyde": hyde}.items():
        paths[n] = tmp_path / f"{n}.json"
        paths[n].write_text(json.dumps(obj), encoding="utf-8")
    outs = {}
    for w in (None, 2.0):
        a = SimpleNamespace(pool=str(paths["pool"]), q8b=str(paths["q8b"]), dense=str(paths["dense"]),
                            hyde=str(paths["hyde"]), out_rrf3d2=str(tmp_path / f"r{w}.json"),
                            out_fhyde=str(tmp_path / f"f{w}.json"), report="", w_q8b_fhyde=w)
        vi23.cmd_fuse(a)
        outs[w] = (json.loads(Path(a.out_rrf3d2).read_text()), json.loads(Path(a.out_fhyde).read_text()))
    assert outs[None][0] == outs[2.0][0]
    assert outs[None][1] != outs[2.0][1]
    _, fh = _chain(2.0)
    assert outs[2.0][1] == fh


def test_coverage_doclist():
    dl = {"1": ["a", "b"], "2": ["c"]}
    q8 = {"1": [["a", 0, 1.0]], "2": [["c", 0, 1.0]]}
    assert vi33.coverage_doclist(dl, q8) == {"cap": 3, "co_8b": 2, "ty_le": 0.6667}


def test_export_name_matches_vi31_glob():
    assert Path("docs_vi31_old_v33.part000.jsonl.zst").match("docs_vi31_old*.jsonl.zst")
