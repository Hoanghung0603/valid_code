"""Tach cands docwin theo tung pool mode (doc trong doclist + top100 cua chinh mode do),
copy docwin.json.zst + docs_docwin parts tu full1200u. Chay tren b660.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, r"D:\r2ai\runner\scripts")
import runner_core as rc  # noqa: E402

ROOT = Path(r"D:\r2ai\data\sub\runner")
U = ROOT / "full1200u"
cands = rc.load_json(U / "cands_docwin.json")
for mode in ("full1200", "full1200b"):
    d = ROOT / mode
    dl = rc.load_json(d / "doclist.json")
    t100 = rc.load_json(d / "top100_bge.json")
    out = {}
    for q, rows in cands.items():
        allowed = {str(x) for x in dl.get(q, [])} | {str(x) for x in t100.get(q, [])}
        out[q] = [r for r in rows if str(r[0]) in allowed]
    rc.write_json(d / "cands_docwin.json", out)
    shutil.copy(U / "docwin.json.zst", d / "docwin.json.zst")
    for f in U.glob("docs_docwin.part*.jsonl.zst"):
        shutil.copy(f, d / f.name)
    print(mode, "cau", len(out), "cap", sum(map(len, out.values())), flush=True)
