r"""q8b36: cham Qwen3-Reranker-8B cho cap thieu diem 8B trong doclist vi36 (luot 1, tinh voi q8b_v33)
+ 30 hang bien theo fhyde_pre. Mot bien so voi q8b33: nguon la doclist/fhyde/pool cua VI36 va
old = q8b_v33; chon cap bang vi36.select_8b_targets (doclist + margin, bo da co diem).

Tai dung q8b31 (select/merge/export text/shard/bench) qua select_pairs thay the.
  select  (b660): chon cap + xuat text/queries/cands shard + bench (cap da co diem q8b_v33).
  merge   (b660/local): q8b_v36 = q8b_v33 + hit moi, sap giam dan; kiem tra + bench cung thang.
Out b660 D:\r2ai\data\sub\vi36\q8b36_in\; goi Modal deploy/modal_q8b36.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import q8b31 as base
import vi36

ROOT = Path("D:/r2ai")
MARGIN = 30
OUT = vi36.OUT
FHYDE = OUT / "cache_fhyde_v36_pre.json"
POOL = OUT / "cache_pool_v36.json"
OLD = vi36.Q8_V33
DOCLIST = OUT / "doclist_v36_pre.json"
NEW_IDS = OUT / "new_ids.json"
INDEX = vi36.IX

_orig_select = base.select_pairs


def select_pairs_doclist(fhyde: dict, pool: dict, q8b: dict) -> dict:
    """{qid: [doc_id...]} = (doclist_pre + margin) - da co diem q8b (vi36.select_8b_targets)."""
    dl = base.rd(DOCLIST)
    targets = vi36.select_8b_targets(dl, fhyde, q8b, MARGIN)
    po = {str(q): {str(h[0]) for h in v} for q, v in pool.items()}
    out = {}
    for q, docs in targets.items():
        keep = [d for d in docs if str(d) in po.get(q, set())]
        if keep:
            out[q] = keep
    return out


def apply_base():
    """Gan cau hinh q8b36 vao q8b31 (goi trong main; bench lay trong top-170 cung thang)."""
    base.PREFIX = "q8b36"
    base.SELECT_FHYDE = FHYDE
    base.SELECT_POOL = POOL
    base.SELECT_Q8B = OLD
    base.SELECT_NEW_IDS = NEW_IDS
    base.IX = INDEX
    base.TOP = 170   # chi dung cho bench + bao cao rank_band


def main():
    apply_base()
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        base.select_pairs = lambda fh, po, q8, top=None: select_pairs_doclist(fh, po, q8)
    base.main()


if __name__ == "__main__":
    main()
