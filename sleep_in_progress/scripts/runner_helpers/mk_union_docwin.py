"""Gop doclist/top100 cua 2 pool mode (p1 = full1200, base = full1200b) vao full1200u de export_docwin 1 lan.

Chay tren b660: python mk_union_docwin.py  (ghi D:\\r2ai\\data\\sub\\runner\\full1200u\\)
pool.json lay tu p1 (superset cua base, cung logit) -> idx tra duoc cho moi doc cua ca 2 mode.
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, r"D:\r2ai\runner\scripts")
import runner_core as rc  # noqa: E402

A = Path(r"D:\r2ai\data\sub\runner\full1200")
B = Path(r"D:\r2ai\data\sub\runner\full1200b")
U = Path(r"D:\r2ai\data\sub\runner\full1200u")
U.mkdir(parents=True, exist_ok=True)
dl = rc.union_pairs(rc.load_json(A / "doclist.json"), rc.load_json(B / "doclist.json"))
t100 = rc.union_pairs(rc.load_json(A / "top100_bge.json"), rc.load_json(B / "top100_bge.json"))
rc.write_json(U / "doclist.json", dl)
rc.write_json(U / "top100_bge.json", t100)
shutil.copy(A / "pool.json", U / "pool.json")
print("union", len(dl), "cau; doc", sum(map(len, dl.values())), "top100", sum(map(len, t100.values())), flush=True)
