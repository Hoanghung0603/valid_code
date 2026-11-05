import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import vi24


def record(d, text):
    return {"id": d, "status": 200, "text": text, "host": "test.vn", "title": "title"}


def test_prep_uses_actual_vi23_ids_and_preserves_empty_bp():
    text = "noi dung y khoa " * 30
    from boilerplate import _norm as norm_line
    kept, bp, rep = vi24.prep_records([record(1, text), record(2, text), record(2, text)],
                                     {1}, {"test.vn": [norm_line(text)]})
    assert [r["id"] for r in kept] == [2]
    assert rep["overlap_vi23"] == 1
    assert rep["new"] == 1
    assert rep["empty_after_boilerplate"] == 1
    assert bp["test.vn"] == [norm_line(text)]


def test_docwin_reuses_old_candidates_outside_doclist_and_known_empty_pairs():
    dl = {"1": [10, 20, 30, 40]}
    old = {"1": [[10, 1, 0, 2.0, "a"], [20, 2, 0, 1.0, "b"]]}
    assert vi24.missing_docwin(dl, old, {"1": [30]}, {"40": 42}) == {"1": [[40, 42]]}


def test_docwin_missing_idx_fails_instead_of_silent_drop():
    with pytest.raises(KeyError):
        vi24.missing_docwin({"1": [1]}, {}, {}, {})


def test_chunk_merge_preserves_old_and_rejects_overlap():
    old = {"1": [[10, 1, 0, 2.0, "old"]]}
    new = {"1": [[20, 1, 0, 3.0, "new"]]}
    out = vi24.merge_chunks(old, new)
    assert out["1"] == new["1"] + old["1"]
    assert old["1"][0][4] == "old"
    with pytest.raises(AssertionError):
        vi24.merge_chunks(old, old)


def test_lock_does_not_remove_another_jobs_lock(tmp_path):
    lock = tmp_path / "heavy.lock"
    lock.write_text("other")
    with pytest.raises(FileExistsError):
        with vi24.owned_lock(lock, "ours"):
            pass
    assert lock.read_text() == "other"
    lock.unlink()
    with pytest.raises(RuntimeError):
        with vi24.owned_lock(lock, "ours"):
            raise RuntimeError("job failed")
    assert not lock.exists()


def test_sanity_and_real_chain_have_vi23_base_and_delta48():
    args = list(map(str, vi24.chain_args(vi24.B, Path("cands"), "km", "khong-moi")))
    assert args[args.index("--pool") + 1] == str(vi24.B / "cache_pool_v23.json")
    assert args[args.index("--delta") + 1] == "48"
    assert args[args.index("--ref-kq") + 1] == str(vi24.REF_K)


def test_guard_retries_then_aborts_without_killing_process(monkeypatch):
    seen = []
    monkeypatch.setattr(vi24, "probe", lambda: False)
    monkeypatch.setattr(vi24.time, "sleep", lambda n: seen.append(n))
    with pytest.raises(RuntimeError, match="cooperatively"):
        vi24.guard_probe()
    assert seen == [30]


@pytest.mark.parametrize("code", [0, 2])
def test_gpu_runner_handles_cli_exit_and_restores_model_loader(monkeypatch, code):
    from types import SimpleNamespace
    class Auto:
        @staticmethod
        def from_pretrained(*a, **kw):
            return None
    original = Auto.from_pretrained
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModel=Auto, AutoModelForSequenceClassification=type("Classifier", (Auto,), {})))
    monkeypatch.setattr(vi24, "guard_probe", lambda: None)
    def cli(*a, **kw):
        raise SystemExit(code)
    monkeypatch.setattr(vi24.runpy, "run_path", cli)
    if code:
        with pytest.raises(SystemExit) as exc:
            vi24.gpu_run("cli.py")
        assert exc.value.code == code
    else:
        vi24.gpu_run("cli.py")
    assert Auto.from_pretrained is original
