import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from merge_pool import check_ref, merge_pool, pool_stats


def test_merge_pool_keeps_old_untouched_and_appends_new():
    old = {"1": [[2, 20, -0.5], [1, 10, 0.25]]}
    new = {"1": [[3, 30, 1.5], [4, 40, None]]}
    out = merge_pool(old, new)
    assert out == {"1": [[2, 20, -0.5], [1, 10, 0.25], [3, 30, 1.5], [4, 40, None]]}
    assert old == {"1": [[2, 20, -0.5], [1, 10, 0.25]]}


def test_merge_pool_skips_docs_already_scored_and_dupes_in_new():
    old = {"1": [[1, 10, 0.1]]}
    new = {"1": [[1, 10, 9.9], [2, 20, 0.2], [2, 20, 0.3], [3, 30, 0.4]]}
    out = merge_pool(old, new)
    assert out == {"1": [[1, 10, 0.1], [2, 20, 0.2], [3, 30, 0.4]]}


def test_merge_pool_takes_new_only_queries():
    out = merge_pool({}, {"9": [[1, 1, 0.5], [1, 1, 0.6]]})
    assert out == {"9": [[1, 1, 0.5]]}


def test_check_ref_accepts_identical_and_reports_breaches():
    merged = {"1": [[1, 10, 0.1], [2, 20, 0.2]]}
    ref = {"1": [[2, 20, 0.2], [1, 10, 0.1]]}
    assert check_ref(merged, ref) == []
    bad = {"1": [[1, 10, 0.1], [1, 10, 0.2], [2, 20, 0.2]]}
    errs = check_ref(bad, ref)
    assert any("trùng" in e for e in errs)
    errs = check_ref({"1": [[1, 10, 0.1]]}, ref)
    assert any("thiếu" in e for e in errs)
    errs = check_ref({"1": [[1, 11, 0.1], [2, 20, 0.2]]}, ref)
    assert any("lệch" in e for e in errs)
    errs = check_ref({"2": []}, ref)
    assert any("lệch bộ câu" in e for e in errs)


def test_pool_stats_before_after_and_new_only():
    old = {"1": [[1, 1, -3.0], [2, 2, -1.0]]}
    merged = {"1": [[1, 1, -3.0], [2, 2, -1.0], [3, 3, -2.0], [4, 4, None]]}
    before = pool_stats(old, old, thr=-2.6)
    after = pool_stats(merged, old, thr=-2.6)
    assert before == {"pool": 2.0, "gt": 1.0, "new_gt": 0.0}
    assert after == {"pool": 4.0, "gt": 2.0, "new_gt": 1.0}


def test_main_writes_sorted_cache_and_stats(tmp_path, capsys, monkeypatch):
    from merge_pool import main
    old = {"1": [[1, 1, -1.0]], "2": [[8, 80, -3.5]]}
    new = {"1": [[2, 2, 0.5], [3, 3, None], [1, 1, 9.9]]}
    ref = {"1": [[1, 1, -1.0]], "2": [[8, 80, -3.5]]}
    p = tmp_path
    (p / "old.json").write_text(json.dumps(old), encoding="utf-8")
    (p / "new.json").write_text(json.dumps(new), encoding="utf-8")
    (p / "ref.json").write_text(json.dumps(ref), encoding="utf-8")
    out, outc = p / "pool2.json", p / "cache2.json"
    monkeypatch.setattr(sys, "argv", [
        "merge_pool.py", "--old", str(p / "old.json"), "--new", str(p / "new.json"),
        "--out", str(out), "--out-cache", str(outc), "--ref", str(p / "ref.json"), "--thr", "0.0",
    ])
    main()
    merged = json.loads(out.read_text(encoding="utf-8"))
    cache = json.loads(outc.read_text(encoding="utf-8"))
    assert merged["1"] == [[1, 1, -1.0], [2, 2, 0.5], [3, 3, None]]
    assert cache["1"] == [[2, 2, 0.5], [1, 1, -1.0], [3, 3, None]]
    assert cache["2"] == [[8, 80, -3.5]]
    txt = capsys.readouterr().out
    assert "kiểm ref OK" in txt
    assert "pool/câu: TB 1.0 → 2.0" in txt
    assert "doc MỚI logit > 0.0/câu: TB 0.5" in txt


def test_main_exits_1_when_ref_broken(tmp_path, monkeypatch):
    from merge_pool import main
    p = tmp_path
    (p / "old.json").write_text(json.dumps({"1": [[1, 1, 0.1]]}), encoding="utf-8")
    (p / "new.json").write_text(json.dumps({"1": [[2, 2, 0.2]]}), encoding="utf-8")
    (p / "ref.json").write_text(json.dumps({"1": [[9, 9, 0.9]]}), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "merge_pool.py", "--old", str(p / "old.json"), "--new", str(p / "new.json"),
        "--out", str(p / "o.json"), "--out-cache", str(p / "c.json"), "--ref", str(p / "ref.json"),
    ])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
