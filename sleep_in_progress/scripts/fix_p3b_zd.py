"""Sua noi dung quy_uoc.zd trong cac report_<name>.json cua p3b (text sinh truoc khi sua mo ta).

Chay: D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\fix_p3b_zd.py
"""
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

W = Path(r"D:\r2ai\data\sub\diag18")
ZD = ("doc co trong cache_rrf3d2 -> z cua cache do (y cx130cov); doc khong co trong rrf3d2 "
      "nhung co trong cache_rrf3d2_p3 (rrf3d2 TAI LAP tren pool3: fuse3 bge + Qwen8B + dense*2, "
      "k=60; 8B vang voi doc NEW nen NEW chi duoc bge+dense dong gop) -> z cua cache moi; con lai "
      "zd = zmin cua cau (diag14). Z tinh rieng tren tung cache theo cau (chunk_docfuse.doc_z)")
KHAC = ("doc nen = doclist_p3_t{20,16}/p3nq (khong phai split); ung vien chunk_dw_u900_p3")

for name in ("submission_p3t20_cx", "submission_p3t16_cx", "submission_p3nq_cx"):
    p = W / f"report_{name}.json"
    rep = json.loads(p.read_text(encoding="utf-8"))
    q = rep.get("quy_uoc", {})
    q["khac"] = KHAC
    q["zd"] = ZD
    q.pop("zd_cu", None)
    rep["quy_uoc"] = q
    p.write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    print(f"sua {p.name}: fused_new={rep.get('fused_new')}", flush=True)
print("done", flush=True)
