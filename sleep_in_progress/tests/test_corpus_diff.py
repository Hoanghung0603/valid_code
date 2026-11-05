import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from corpus_diff import diff_corpus, sub_membership  # noqa: E402


OLD = {1: "https://a.vn/1", 2: "https://a.vn/2", 3: "https://b.cn/3", 4: "https://b.cn/4"}
NEW = {1: "https://a.vn/1", 2: "https://a.vn/2x", 3: "https://b.cn/3", 9: "https://pubmed.ncbi.nlm.nih.gov/9/"}


def test_diff_counts():
    c = diff_corpus(OLD, NEW)
    assert c["old_n"] == 4 and c["new_n"] == 4
    assert c["kept"] == 3
    assert c["kept_url_changed"] == 1
    assert c["dropped"] == 1 and c["added"] == 1
    assert c["added_url_was_in_old"] == 0
    hosts = {h["host"]: h for h in c["by_host"]}
    assert hosts["b.cn"]["old"] == 2 and hosts["b.cn"]["new"] == 1 and hosts["b.cn"]["kept"] == 1
    assert hosts["pubmed.ncbi.nlm.nih.gov"]["old"] == 0


def test_sub_membership():
    sub = [
        {"id": 1, "relevant_docs": [1, 4], "relevant_chunks": [{"doc_id": 1, "chunk_text": "x"}, {"doc_id": 4, "chunk_text": "y"}]},
        {"id": 2, "relevant_docs": [3], "relevant_chunks": []},
    ]
    s = sub_membership(sub, set(NEW), OLD)
    assert s["doc_avg_per_q"] == 1.5
    assert abs(s["doc_in_new_frac"] - 2 / 3) < 1e-9
    assert abs(s["macro_in_frac"] - 0.75) < 1e-9
    assert s["chunk_in_new_frac"] == 0.5
    assert s["out_hosts_top"] == [("b.cn", 1)]
    assert s["by_rank30"]["1-30"] == [3, 2, 0.6667]
