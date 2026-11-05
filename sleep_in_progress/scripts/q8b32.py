"""Thin q8b32 config: Qwen3-Reranker-8B for missing pairs in top-200 vi31."""
from pathlib import Path

import q8b31 as base


ROOT = Path("D:/r2ai")
TOP = 170
OUT = ROOT / "data/sub/diag32"
FHYDE = ROOT / "data/sub/vi31/cache_fhyde_v31.json"
POOL = ROOT / "data/sub/vi31/in1/pool_v31.json"
OLD = ROOT / "data/sub/vi31/q8b_v31_clean.json"
NEW_IDS = ROOT / "data/sub/diag30/new_ids.json"
INDEX = ROOT / "data/index/vi30"

base.TOP = TOP
base.PREFIX = "q8b32"
base.SELECT_FHYDE = FHYDE
base.SELECT_POOL = POOL
base.SELECT_Q8B = OLD
base.SELECT_NEW_IDS = NEW_IDS
base.IX = INDEX


def main():
    base.main()


if __name__ == "__main__":
    main()
