import json
import zipfile

from medret.schemas import ChunkRef, QueryResult
from medret.submit import (
    iter_submission,
    load_results,
    pack_submission,
    read_submission,
    submission_exists,
    validate_submission,
    write_submission,
)


def results():
    return [
        QueryResult(2, ["5"], [ChunkRef("5", "Đoạn văn tiếng Việt")]),
        QueryResult(1, [], []),
    ]


def test_write_roundtrip_and_validate(tmp_path):
    path = write_submission(results(), tmp_path / "sub.json", doc_id_type="int")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [d["id"] for d in data] == [1, 2]
    assert data[1]["relevant_docs"] == [5] and data[1]["relevant_chunks"][0]["doc_id"] == 5
    assert "Đoạn văn" in path.read_text(encoding="utf-8")  # not \u-escaped
    assert validate_submission(path, query_ids=[1, 2], corpus_ids={"5"}) == []
    assert [r.id for r in load_results(path)] == [1, 2]


def test_str_doc_ids(tmp_path):
    path = write_submission(results(), tmp_path / "sub.json", doc_id_type="str")
    assert json.loads(path.read_text(encoding="utf-8"))[1]["relevant_docs"] == ["5"]


def test_validate_reports_problems(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(
        json.dumps([
            {"id": 1, "relevant_docs": [9]},
            {
                "id": 1,
                "relevant_docs": [],
                "relevant_chunks": [{"doc_id": 5, "chunk_order": -1, "chunk_text": "x"}],
            },
        ]),
        encoding="utf-8",
    )
    errors = validate_submission(path, query_ids=[1, 2], corpus_ids={"5"})
    text = "\n".join(errors)
    assert "relevant_chunks must be a list" in text
    assert "doc 9 not in links_corpus" in text
    assert "duplicate id 1" in text
    assert "chunk_order" in text
    assert "1 query ids missing" in text


def test_pack_single_file_at_root(tmp_path):
    path = write_submission(results(), tmp_path / "sub.json")
    with zipfile.ZipFile(pack_submission(path)) as zf:
        assert zf.namelist() == ["sub.json"]


def test_pack_big_dictionary_and_reorder_roundtrip(tmp_path):
    import json
    import zipfile

    from medret.submit.submission import pack_submission

    items = [{"id": i, "relevant_docs": [str(i % 3)],
              "relevant_chunks": [{"doc_id": str(i % 3), "chunk_text": "văn bản lặp lại " * 50}]} for i in range(6)]
    src = tmp_path / "submission.json"
    src.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    z = pack_submission(src, tmp_path / "s.zip", dict_mb=64, reorder=True)
    with zipfile.ZipFile(z) as zf:  # the standard reader decodes the custom dictionary size
        assert zf.namelist() == ["submission.json"] and zf.infolist()[0].compress_type == zipfile.ZIP_LZMA
        back = json.loads(zf.read("submission.json"))
    assert sorted(x["id"] for x in back) == list(range(6))
    assert {x["id"]: x for x in back} == {x["id"]: x for x in items}
    assert zipfile.LZMACompressor.__name__ == "LZMACompressor"  # the patch is undone


def test_read_from_zip_after_json_deleted(tmp_path):
    path = write_submission(results(), tmp_path / "submission.json")
    pack_submission(path, tmp_path / "run.zip")
    path.unlink()
    assert submission_exists(path)
    assert [r.id for r in load_results(path)] == [1, 2]
    assert read_submission(path)[0]["id"] == 1
    assert not submission_exists(tmp_path / "other.json")


def test_iter_submission_matches_read_submission(tmp_path):
    items = [{"id": 1, "relevant_docs": [1, 2], "relevant_chunks": [{"doc_id": 1, "chunk_text": 'a ] , [ "x"'}]},
             {"id": 2, "relevant_docs": [], "relevant_chunks": []}]
    p = tmp_path / "s.json"
    for text in ("[" + ", ".join(json.dumps(x) for x in items) + "]", json.dumps(items, indent=2)):
        p.write_text(text, encoding="utf-8")
        assert list(iter_submission(p)) == read_submission(p) == items
        for block in (1, 3, 7):  # entries and separators split across read blocks
            assert list(iter_submission(p, block_chars=block)) == items
    p.write_text("[]", encoding="utf-8")
    assert list(iter_submission(p)) == []


def test_iter_submission_reads_from_the_zip(tmp_path):
    items = [{"id": 1, "relevant_docs": [3], "relevant_chunks": [{"doc_id": 3, "chunk_text": "Đoạn văn"}]}]
    with zipfile.ZipFile(tmp_path / "run.zip", "w") as zf:
        zf.writestr("submission.json", json.dumps(items, ensure_ascii=False))
    assert list(iter_submission(tmp_path / "submission.json", block_chars=5)) == items
