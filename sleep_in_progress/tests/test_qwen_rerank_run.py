import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from qwen_rerank_run import (  # noqa: E402
    PREFIX,
    SUFFIX,
    StopJob,
    Watchdog,
    build_input_ids,
    build_pairs,
    score_stats,
)
from rerank_run import merge_scores  # noqa: E402


class CharTok:
    """Tokenizer giả: mỗi ký tự = 1 token (mã ord)."""

    def __call__(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


def ids(s):
    return [ord(c) for c in s]


INSTR = "judge"


def test_build_input_ids_prefix_suffix_intact_and_short_doc_untouched():
    tok = CharTok()
    out = build_input_ids(tok, INSTR, "q?", "doc ngắn", max_len=1024)
    assert out[: len(PREFIX)] == ids(PREFIX)
    assert out[-len(SUFFIX):] == ids(SUFFIX)
    body = f"<Instruct>: {INSTR}\n<Query>: q?\n<Document>: doc ngắn"
    assert out == ids(PREFIX + body + SUFFIX)
    assert len(out) <= 1024


def test_build_input_ids_long_doc_truncated_to_max_len():
    tok = CharTok()
    doc = "x" * 5000
    out = build_input_ids(tok, INSTR, "hỏi", doc, max_len=400)
    assert len(out) == 400
    assert out[: len(PREFIX)] == ids(PREFIX)
    assert out[-len(SUFFIX):] == ids(SUFFIX)
    head = ids(PREFIX + f"<Instruct>: {INSTR}\n<Query>: hỏi\n<Document>: ")
    assert out[: len(head)] == head
    assert set(out[len(head): -len(SUFFIX)]) == {ord("x")}


def test_build_input_ids_long_query_cut_to_256():
    tok = CharTok()
    q = "q" * 600
    out = build_input_ids(tok, INSTR, q, "d" * 3000, max_len=1024)
    assert len(out) <= 1024
    s = "".join(chr(i) for i in out)
    assert ("<Query>: " + "q" * 256 + "\n<Document>: ") in s
    assert "q" * 257 not in s
    assert s.startswith(PREFIX) and s.endswith(SUFFIX)
    # max_query_tokens tuỳ chỉnh
    out2 = build_input_ids(tok, INSTR, q, "", max_len=2048, max_query_tokens=10)
    assert "<Query>: " + "q" * 10 + "\n<Document>: " in "".join(chr(i) for i in out2)


def test_merge_scores_order_and_missing_text_none():
    docs = {"a": "ta", "c": "tc"}
    cand = [["a", 1], ["b", 2], ["c", 3], ["d", 4]]
    pairs, pos = build_pairs("qq", cand, docs, top=3)
    assert pairs == [("qq", "ta"), ("qq", "tc")] and pos == [0, 2]
    res = merge_scores(cand, 3, pos, [1.234567, -2.0])
    assert res == [["a", 1, 1.2346], ["b", 2, None], ["c", 3, -2.0]]


def test_score_stats():
    st = score_stats([None, 3.0, 1.0, 2.0])
    assert st["min"] == 1.0 and st["max"] == 3.0 and st["p50"] == 2.0
    assert score_stats([None]) == {}


# ---------------------------------------------------------------- watchdog

class Fake:
    """Đồng hồ + sleep + measure giả cho Watchdog."""

    def __init__(self, lats=None, temps=None):
        self.t = 0.0
        self.sleeps = []
        self.lats = list(lats or [])
        self.temps = list(temps or [])
        self.n_measure = 0
        self.logs = []

    def time(self):
        return self.t

    def sleep(self, s):
        self.sleeps.append(s)
        self.t += s

    def measure(self):
        self.n_measure += 1
        v = self.lats.pop(0) if self.lats else 0.2
        if isinstance(v, Exception):
            raise v
        return v

    def temp(self):
        return self.temps.pop(0) if self.temps else 60

    def wd(self, **kw):
        return Watchdog(self.measure, self.temp, sleep_fn=self.sleep, time_fn=self.time,
                        log=self.logs.append, **kw)

    def run(self, wd, n, step=61):
        for _ in range(n):
            wd.check()
            self.t += step


def test_watchdog_normal_no_sleep():
    f = Fake(lats=[0.2, 0.5, 0.9, 1.0])
    f.run(f.wd(), 4)
    assert f.sleeps == [] and f.n_measure == 4


def test_watchdog_three_slow_pause_once():
    f = Fake(lats=[2.0, 2.0, 2.0, 0.2, 0.2])
    f.run(f.wd(), 5)
    assert f.sleeps == [300]


def test_watchdog_slow_interleaved_no_pause():
    f = Fake(lats=[2.0, 2.0, 0.2, 2.0, 2.0, 0.3, 2.0])
    f.run(f.wd(), 7)
    assert f.sleeps == []


def test_watchdog_too_many_pauses_raises_stop():
    f = Fake(lats=[5.0] * 100)
    wd = f.wd()
    with pytest.raises(StopJob):
        f.run(wd, 100, step=61)  # mỗi pause cách 3×61+300=483s -> 7 lần trong 2898s < 3600s
    assert f.sleeps == [300] * 6
    assert any("STOP latency" in m for m in f.logs)


def test_watchdog_pauses_outside_window_not_counted():
    f = Fake(lats=[5.0] * 60)
    wd = f.wd()
    f.run(wd, 60, step=400)  # mỗi pause cách nhau > 3×400+300 s -> tối đa 3 lần/giờ
    assert len(f.sleeps) == 20


def test_watchdog_hot_gpu_sleeps_120():
    f = Fake(temps=[85])
    f.run(f.wd(), 1)
    assert f.sleeps == [120]
    f2 = Fake(temps=[83, None])
    f2.run(f2.wd(), 2)
    assert f2.sleeps == []


def test_watchdog_measures_once_within_interval():
    f = Fake()
    wd = f.wd()
    for _ in range(10):
        wd.check()
        f.t += 5  # tổng 50s < 60s
    assert f.n_measure == 1
    f.t = 61
    wd.check()
    assert f.n_measure == 2


def test_watchdog_measure_exception_counts_as_slow():
    f = Fake(lats=[RuntimeError("timeout"), None, OSError("dns")])
    f.run(f.wd(), 3)
    assert f.sleeps == [300]


def test_import_without_torch(monkeypatch):
    import importlib
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "transformers", None)
    import qwen_rerank_run
    importlib.reload(qwen_rerank_run)
    assert qwen_rerank_run.PART_SIZE == 50
