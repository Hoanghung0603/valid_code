import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from embed_all import load_done, merge_topk, next_block_index


def test_merge_topk():
    best_s = np.full((2, 3), -np.inf, dtype=np.float32)
    best_i = np.full((2, 3), -1, dtype=np.int64)
    merge_topk(best_s, best_i, np.array([0.5, 0.1, 0.9], np.float32), np.arange(3), 0)
    assert list(best_i[0]) == [2, 0, 1]
    assert best_s[0][0] == np.float32(0.9)
    merge_topk(best_s, best_i, np.array([0.7, 0.2], np.float32), np.array([10, 11]), 0)
    assert list(best_i[0]) == [2, 10, 0]
    assert list(best_i[1]) == [-1, -1, -1]


def test_load_done_and_next_block(tmp_path):
    old, new = tmp_path / "emb", tmp_path / "emb_all"
    old.mkdir()
    new.mkdir()
    np.savez(old / "blk_00000.npz", ids=np.array([1, 2], np.int64), emb=np.zeros((2, 4), np.float16))
    np.savez(new / "blk_00005.npz", ids=np.array([3], np.int64), emb=np.zeros((1, 4), np.float16))
    assert load_done([old, new]) == {1, 2, 3}
    assert next_block_index(new) == 6
    assert next_block_index(tmp_path / "missing") == 0
