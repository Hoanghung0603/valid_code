import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "vi23_chunkcheck", Path(__file__).parents[1] / "deploy" / "vi23_chunkcheck.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def test_reused_candidates_require_text_and_windows():
    cands = {"1": [[10, 0], [20, 1]]}
    wins = {"1": {"10": [[0, 4, 1.0]], "20": [[0, 4, 2.0]]}}
    assert check.coverage(cands, wins, {10: "old"}) == {
        "pairs": 2, "missing_text": 1, "missing_wins": 0, "empty_text_pairs": 0, "ok": False}
    assert check.coverage(cands, wins, {10: "old", 20: "reuse"})["ok"]
    del wins["1"]["20"]
    assert not check.coverage(cands, wins, {10: "old", 20: "reuse"})["ok"]


def test_empty_cleaned_text_needs_no_windows():
    result = check.coverage({"1": [[10, 0]]}, {"1": {}}, {10: ""})
    assert result["ok"] and result["empty_text_pairs"] == 1
