import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "deploy"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import crawl_recovery_plan as recovery  # noqa: E402
from crawl import HARD_HOSTS, MAX_ATTEMPTS, TIER2_HOSTS, State, host_of, tier_of  # noqa: E402


def inputs(tmp_path, rows, logs):
    corpus = tmp_path / "links.parquet"
    pd.DataFrame(rows, columns=["id", "url"]).to_parquet(corpus)
    state = tmp_path / "crawl" / "state"
    state.mkdir(parents=True)
    for name, content in logs.items():
        (state / name).write_bytes(content.encode("utf-8"))
    return corpus, state, state.parent / "recovery"


def candidates(out):
    return [json.loads(line) for line in (out / "candidates.jsonl").read_text(encoding="utf-8").splitlines()]


def test_normalization_and_tier_tables_match_crawler():
    assert recovery.HARD_HOSTS == HARD_HOSTS
    assert recovery.TIER2_HOSTS == TIER2_HOSTS
    assert recovery.MAX_ATTEMPTS == MAX_ATTEMPTS
    for host in HARD_HOSTS | TIER2_HOSTS | {
        "vinmec.com", "hellobacsi.com", "suckhoeviet.org.vn", "benhvienvietduc.org",
        "laodong.vn", "zhongyibaodian.net", "sub.cnkang.com", "cnkang.com:443",
    }:
        for url in (f"https://WWW.{host}/x", f"https://{host}/x"):
            normalized = recovery.host_of(url)
            assert normalized == host_of(url)
            assert recovery.tier_of(normalized) == tier_of(normalized)


def test_exact_terminal_semantics_and_retry_cutoff(tmp_path):
    rows = [(i, f"https://cnkang.com/{i}") for i in range(1, 10)]
    corpus, state, out = inputs(tmp_path, rows, {
        "log-00001.tsv": "1\ttransient:TimeoutError\n2\ttransient\n3\ttransient\n"
                         "4\ttransient\n5\tok\n6\tpermanent\n7\tok:Error\n8\tpermanent:Error\n",
        "log-00000.tsv": "2\ttransient\n3\ttransient\n4\ttransient\n",
        "log-00002.tsv": "3\ttransient\n4\ttransient\n4\ttransient\n5\ttransient\n6\ttransient\n",
    })
    report = recovery.generate_plan(corpus, state, out)
    exported = candidates(out)
    assert [r["id"] for r in exported] == [1, 2]
    assert exported[1]["nonterminal_attempts"] == exported[1]["total_attempts"] == 2
    assert all(r["requires_approval"] and r["terminal_state"] is None for r in exported)
    assert report["totals"]["urls"] == 9
    assert report["totals"]["never_attempted"] == 1
    assert report["hosts"]["cnkang.com"]["reasons"] == {
        "propose_transient_retry": 2, "archive_retry_limit": 2, "archive_already_ok": 1,
        "archive_permanent": 1, "archive_other_outcome": 2, "archive_never_attempted": 1,
    }
    # State construction appends a log: use a separate copy solely for semantic parity.
    reference = tmp_path / "reference"
    reference.mkdir()
    for log in state.glob("log-*.tsv"):
        (reference / log.name).write_bytes(log.read_bytes())
    crawler_state = State(reference)
    history, _ = recovery.read_state(state)
    for sid, h in history.items():
        assert h.nonterminal_attempts == crawler_state.attempts[sid]
        assert (h.terminal_state is not None) == (sid in crawler_state.done)
    crawler_state.close()


def test_malformed_and_truncated_lines_preserve_history(tmp_path):
    corpus, state, out = inputs(tmp_path, [(i, f"https://zydcd.com/{i}") for i in range(1, 5)], {
        "log-00000.tsv": "1\tpermanent\n2\tok\n3\ttransient\n",
        "log-00001.tsv": "\nno-tab\nbad\tok\n1\t\n2\t   \n1\tok\textra\n"
                         "1\ttransient\n2\ttransient\n3\ttransient\n4\tok",
    })
    with (state / "log-00002.tsv").open("wb") as fh:
        fh.write(b"1\t\xff\n")
    report = recovery.generate_plan(corpus, state, out)
    diag = report["state_diagnostics"]
    assert diag["empty_lines"] == 1
    assert diag["malformed_lines"] == 4
    assert diag["blank_outcomes"] == 2
    assert diag["truncated_lines"] == 1
    assert [(r["id"], r["reason"]) for r in candidates(out)] == [
        (1, "inspect_short_content"), (3, "propose_transient_retry")]
    first = candidates(out)[0]
    assert first["last_outcome"] == "transient"
    assert first["terminal_state"] == "permanent"
    assert first["nonterminal_attempts"] == 1 and first["total_attempts"] == 2


def test_host_blocking_beats_url_retry_and_inspection(tmp_path):
    corpus, state, out = inputs(tmp_path, [
        (1, "https://www.cnkang.com/a"), (2, "https://cnkang.com/b"),
        (3, "https://zysjonline.com/c"), (4, "https://zydcd.com/d"),
        (5, "https://www.zydcd.com/e"), (6, "https://other.cn/f"),
    ], {"log-00000.tsv": "1\tblocked:Error\n1\tok\n2\ttransient:TimeoutError\n"
                         "3\ttransient\n4\tpermanent\n5\tblocked\n6\ttransient\n"
                         "999\tblocked\n"})
    report = recovery.generate_plan(corpus, state, out)
    assert [r["id"] for r in candidates(out)] == [6]
    assert report["blocked_hosts_all_corpus"] == ["cnkang.com", "zydcd.com"]
    assert report["state_diagnostics"]["ids_absent_from_corpus"] == 1
    assert report["totals"]["archived"] == 5
    assert report["hosts"]["zysjonline.com"]["hard_host"] is True


def test_inspection_paths_and_terminal_history(tmp_path):
    hosts = list(recovery.INSPECTION_HOSTS) + ["qdnd.vn", "other.cn", "zydcd.com", "qdnd.vn"]
    corpus, state, out = inputs(tmp_path, [(i, f"https://{host}/{i}") for i, host in enumerate(hosts, 1)], {
        "log-00000.tsv": "".join(f"{i}\tpermanent\n{i}\ttransient:TimeoutError\n" for i in range(1, 7))
                         + "7\ttransient:TooManyRedirects\n7\ttransient:TooManyRedirects\n"
                         + "7\ttransient:TooManyRedirects\n8\tpermanent\n8\ttransient\n"
                         + "9\tpermanent\n9\tok\n9\ttransient\n10\ttransient:TimeoutError\n",
    })
    recovery.generate_plan(corpus, state, out)
    exported = candidates(out)
    assert [r["id"] for r in exported] == [1, 2, 3, 4, 5, 6, 7, 10]
    assert [r["reason"] for r in exported[:6]] == list(recovery.INSPECTION_HOSTS.values())
    assert exported[6]["reason"] == "inspect_redirects"
    assert exported[6]["nonterminal_attempts"] == 3
    assert exported[-1]["reason"] == "propose_transient_retry"
    assert all(r["requires_approval"] is True for r in exported)


def test_exact_host_exclusions_and_tier_filters(tmp_path):
    corpus, state, out = inputs(tmp_path, [
        (1, "https://www.cnkang.com/1"), (2, "https://sub.cnkang.com/2"),
        (3, "https://notcnkang.com/3"), (4, "https://120ask.com/4"),
        (5, "https://laodong.vn/5"), (6, "https://zysjonline.com/6"),
    ], {"log-00000.tsv": "".join(f"{i}\ttransient\n" for i in range(1, 7))})
    report = recovery.generate_plan(corpus, state, out, hosts="cnkang.com,120ask.com,laodong.vn",
                                    exclude_hosts="120ask.com", tiers="2,3")
    assert [r["id"] for r in candidates(out)] == [1]
    assert report["totals"]["urls"] == 1
    assert list(report["hosts"]) == ["cnkang.com"]
    assert report["corpus_rows"] == 6


def test_missing17_override_ok_state_and_cli(tmp_path, capsys):
    corpus, state, out = inputs(tmp_path, [(i, f"https://120ask.com/{i}") for i in range(1, 20)], {
        "log-00000.tsv": "".join(f"{i}\tok\n" for i in range(1, 20)),
    })
    summary = state.parent / "b660verify_summary.json"
    summary.write_text(json.dumps({"missing": 17, "missing_samples": list(range(1, 18))}), encoding="utf-8")
    before = {p: p.read_bytes() for p in (corpus, summary, *state.glob("log-*.tsv"))}
    assert recovery.main(["--corpus", str(corpus), "--state", str(state), "--out", str(out),
                          "--missing-summary", str(summary), "--hosts", "120ask.com", "--tiers", "2"]) == 0
    assert "candidates: 17" in capsys.readouterr().out
    exported = candidates(out)
    assert [r["id"] for r in exported] == list(range(1, 18))
    assert all(r["reason"] == "recover_missing_shard_record" and r["terminal_state"] == "ok"
               and r["requires_approval"] is True for r in exported)
    assert all(p.read_bytes() == data for p, data in before.items())
    report = json.loads((out / "inventory.json").read_text(encoding="utf-8"))
    assert report["totals"]["archived"] == 2
    assert report["missing_summary"] == {"count": 17, "in_scope": 17, "excluded_by_filters": 0}


@pytest.mark.parametrize("summary", [
    {"missing": 17, "missing_samples": [1, 2]},
    {"missing": 2, "missing_samples": [1, 1]},
    {"missing": 1, "missing_samples": [999]},
    {"missing": 1, "missing_samples": ["1"]},
    {"missing": True, "missing_samples": [1]},
])
def test_missing_summary_refuses_partial_or_unmatched_export(tmp_path, summary):
    corpus, state, out = inputs(tmp_path, [(1, "https://120ask.com/1")], {})
    source = state.parent / "b660verify_summary.json"
    source.write_text(json.dumps(summary), encoding="utf-8")
    with pytest.raises(ValueError):
        recovery.generate_plan(corpus, state, out, missing_summary=source)
    assert not out.exists()


def test_missing_filters_and_blocked_host_still_apply(tmp_path):
    corpus, state, out = inputs(tmp_path, [
        (1, "https://cnkang.com/1"), (2, "https://120ask.com/2"),
        (3, "https://120ask.com/3"), (4, "https://other.cn/4"),
    ], {"log-00000.tsv": "1\tok\n2\tok\n3\tblocked\n4\tok\n"})
    source = state.parent / "b660verify_summary.json"
    source.write_text(json.dumps({"missing": 3, "missing_samples": [1, 2, 4]}), encoding="utf-8")
    report = recovery.generate_plan(corpus, state, out, exclude_hosts="other.cn", tiers="2", missing_summary=source)
    assert [r["id"] for r in candidates(out)] == [1]
    assert report["missing_summary"] == {"count": 3, "in_scope": 2, "excluded_by_filters": 1}
    assert report["hosts"]["120ask.com"]["reasons"] == {"archive_host_blocked": 2}


@pytest.mark.parametrize("name", recovery.OUTPUT_NAMES)
def test_refuses_existing_named_output_without_touching_inputs(tmp_path, name):
    corpus, state, out = inputs(tmp_path, [(1, "https://cnkang.com/1")], {"log-00000.tsv": "1\ttransient\n"})
    out.mkdir()
    existing = out / name
    existing.write_bytes(b"user data")
    with pytest.raises(FileExistsError):
        recovery.generate_plan(corpus, state, out)
    assert existing.read_bytes() == b"user data"
    assert sorted(p.name for p in out.iterdir()) == [name]


def test_output_guards_and_input_containment(tmp_path):
    corpus, state, out = inputs(tmp_path, [(1, "https://cnkang.com/1")], {})
    for unsafe in (state, state / "nested", state.parent, tmp_path / "outside", corpus):
        with pytest.raises(ValueError):
            recovery.generate_plan(corpus, state, unsafe)
    summary = out / "summary.json"
    out.mkdir()
    summary.write_text('{"missing":0,"missing_samples":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="input file"):
        recovery.generate_plan(corpus, state, out, missing_summary=summary)
    assert not (out / "inventory.json").exists()
