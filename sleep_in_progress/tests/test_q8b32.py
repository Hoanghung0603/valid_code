import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import q8b32  # noqa: E402
from q8b31 import rank_band_dist  # noqa: E402


def test_q8b32_uses_vi31_inputs_top200_and_vi30_index():
    assert q8b32.TOP == 170
    assert q8b32.FHYDE.as_posix() == "D:/r2ai/data/sub/vi31/cache_fhyde_v31.json"
    assert q8b32.POOL.as_posix() == "D:/r2ai/data/sub/vi31/in1/pool_v31.json"
    assert q8b32.OLD.as_posix() == "D:/r2ai/data/sub/vi31/q8b_v31_clean.json"
    assert q8b32.NEW_IDS.as_posix() == "D:/r2ai/data/sub/diag30/new_ids.json"
    assert q8b32.INDEX.as_posix() == "D:/r2ai/data/index/vi30"


def test_q8b32_rank_bands_count_scored_new_pairs():
    fhyde = {"1": [[d, 0, 1.0] for d in range(1, 201)]}
    new = {"1": [[1, 0, 2.0], [100, 0, 2.0], [101, 0, 2.0],
                  [150, 0, 2.0], [151, 0, 2.0], [200, 0, None], [201, 0, 2.0]]}
    assert rank_band_dist(fhyde, new, top=200) == {
        "1-100": 2, "101-150": 2, "151-200": 1, "ngoai_top": 1
    }


def test_q8b32_selection_includes_ranks_through_170():
    fhyde = {"1": [[d, 0, 1.0] for d in range(1, 201)]}
    pool = {"1": [[170, 0, 0]]}
    assert q8b32.base.select_pairs(fhyde, pool, {}, q8b32.TOP) == {"1": [170]}
