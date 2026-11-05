r"""q8b33: cham Qwen3-Reranker-8B cho cap thieu diem trong TOAN doclist vi33 (+ bien MARGIN hang).

Khac q8b32 (top-170 co dinh): moi cau lay top-(len(doclist_vi33[q]) + MARGIN) cua fhyde_v33
(w_q8b = 2), vi n_q toi 400 vuot vung 170 da bu. MARGIN de doc ngay duoi duong cat, khi co
diem 8B, co the vao doclist (khong thi doc thieu nguon bi thiet co he thong).
Tai dung q8b31 (select/merge, export text, shard, bench) qua select_pairs thay the.
Out b660 D:\r2ai\data\sub\diag33\. Artefact vi33/diag32 CHI DOC.
"""
from pathlib import Path

import q8b31 as base

ROOT = Path("D:/r2ai")
MARGIN = 30
OUT = ROOT / "data/sub/diag33"
FHYDE = ROOT / "data/sub/vi33/cache_fhyde_v31.json"
POOL = ROOT / "data/sub/vi33/in1/pool_v31.json"
OLD = ROOT / "data/sub/vi33/q8b_v31_clean.json"
DOCLIST = ROOT / "data/sub/vi33/doclist_vi33.json"
NEW_IDS = ROOT / "data/sub/diag30/new_ids.json"
INDEX = ROOT / "data/index/vi30"

_orig_select = base.select_pairs


def per_query_top(doclist: dict, margin: int = MARGIN) -> dict:
    """{qid: len(doclist[q]) + margin}."""
    return {str(q): len(v) + margin for q, v in doclist.items()}


def select_pairs_doclist(fhyde: dict, pool: dict, q8b: dict, tops: dict) -> dict:
    """Nhu q8b31.select_pairs nhung top rieng tung cau (cau khong co trong tops -> 0 cap)."""
    out = {}
    for q, hits in fhyde.items():
        k = tops.get(str(q), 0)
        out.update(_orig_select({q: hits}, pool, q8b, k))
    return out


def configure(doclist: dict, margin: int = MARGIN):
    """Gan base.select_pairs = ban theo doclist; bench van lay trong top-170 (cung thang)."""
    tops = per_query_top(doclist, margin)
    base.select_pairs = lambda fh, po, q8, top=None: select_pairs_doclist(fh, po, q8, tops)
    return tops


def apply_base():
    """Gan cau hinh q8b33 vao q8b31 (goi trong main, khong o muc import: q8b32 cung gan base)."""
    base.PREFIX = "q8b33"
    base.SELECT_FHYDE = FHYDE
    base.SELECT_POOL = POOL
    base.SELECT_Q8B = OLD
    base.SELECT_NEW_IDS = NEW_IDS
    base.IX = INDEX
    base.TOP = 170   # chi dung cho bench + bao cao rank_band


def main():
    import sys
    apply_base()
    if len(sys.argv) > 1 and sys.argv[1] == "select":
        configure(base.rd(DOCLIST))
    base.main()


if __name__ == "__main__":
    main()
