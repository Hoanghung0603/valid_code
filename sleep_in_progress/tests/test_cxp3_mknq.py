import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cxp3 import mknq, swapdoc  # noqa: E402
from test_cxp3 import FU3, _ref_cx130, _rows3  # noqa: E402


def test_mknq_top_n_q_theo_thu_hang_va_thong_ke(tmp_path):
    # hang: 30 > 10 > 40 > 11; split cau 1 co 2 doc [10, 11] -> top-2 = [30, 10]
    rank = {"1": [[30, 1, 3.0], [10, 2, 2.0], [40, 3, 1.0], [11, 4, 0.5]],
            "2": [[20, 1, 1.0]]}
    split = [{"id": 1, "relevant_docs": [10, 11], "relevant_chunks": []},
             {"id": 2, "relevant_docs": [20], "relevant_chunks": []}]
    new = {"1": [30], "2": []}
    for name, obj in (("rank.json", rank), ("split.json", split), ("new.json", new)):
        (tmp_path / name).write_text(json.dumps(obj), encoding="utf-8")
    out, rep = tmp_path / "dl.json", tmp_path / "rep.json"
    r = mknq(str(tmp_path / "rank.json"), str(tmp_path / "split.json"), str(tmp_path / "new.json"),
             str(out), str(rep))
    dl = json.loads(out.read_text(encoding="utf-8"))
    assert dl == {"1": [30, 10], "2": [20]}
    assert r["TB_NEW_lot_cau"] == 0.5  # cau 1 co 1 NEW (30), cau 2 co 0
    assert r["TB_doc_day_ra_cau"] == 0.5  # cau 1 mat doc 11, cau 2 khong mat
    assert r["n_q_TB"] == 1.5 and r["TB_doc_cau"] == 1.5
    assert json.loads(rep.read_text(encoding="utf-8"))["tong_NEW_lot"] == 1


def test_mknq_bo_trung_va_thieu_hang(tmp_path):
    rank = {"1": [[30, 1, 3.0], [30, 2, 2.0], [10, 3, 1.0]]}  # trung doc 30
    split = [{"id": 1, "relevant_docs": [10, 11, 99], "relevant_chunks": []}]  # n_q = 3 > so hang
    (tmp_path / "rank.json").write_text(json.dumps(rank), encoding="utf-8")
    (tmp_path / "split.json").write_text(json.dumps(split), encoding="utf-8")
    r = mknq(str(tmp_path / "rank.json"), str(tmp_path / "split.json"), "",
             str(tmp_path / "dl.json"), str(tmp_path / "rep.json"))
    dl = json.loads((tmp_path / "dl.json").read_text(encoding="utf-8"))
    assert dl == {"1": [30, 10]}  # bo trung, khong du 3 thi lay het
    assert r["min_doc"] == 2 and r["TB_NEW_lot_cau"] == 0.0  # khong truyen --new


def test_swapdoc_dem_new_cau(tmp_path):
    ref = _ref_cx130(_rows3(), FU3, {"10", "30"}, 2)
    cx = tmp_path / "cx.json"
    cx.write_text(json.dumps(ref), encoding="utf-8")
    dl = tmp_path / "dl.json"
    dl.write_text(json.dumps({"1": [30]}), encoding="utf-8")  # thiếu doc 10 -> nối
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"1": [30]}), encoding="utf-8")
    rep = swapdoc(str(cx), {"sub_n": str(dl)}, str(tmp_path / "out"), str(new))
    st = rep["bai"]["sub_n"]
    assert st["TB_NEW_cau"] == 1.0  # doc 30 NEW, doc 10 (nối) không NEW
    assert st["TB_doc_noi_cau"] == 1.0
    rep2 = swapdoc(str(cx), {"sub_n": str(dl)}, str(tmp_path / "out2"))
    assert rep2["bai"]["sub_n"]["TB_NEW_cau"] is None
