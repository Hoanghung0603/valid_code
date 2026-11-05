import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from diag31 import build_docceil30, cut_chunks, zip_lzma  # noqa: E402


def _sub(chunks=None, docs=None):
    return [{"id": 1, "relevant_docs": docs if docs is not None else [6],
             "relevant_chunks": chunks if chunks is not None else []}]


def test_build_docceil30_fhyde_first_then_pool_then_vi30():
    pool = {"1": [[5, 0, 1.0], [6, 0, 0.5], [7, 0, 0.0]]}
    fhyde = {"1": [[7, 0, 0.9], [9, 0, 0.8], [6, 0, 0.7]]}
    out, a0, a1 = build_docceil30(_sub(docs=[6, 8]), pool, fhyde)
    assert out[0]["relevant_docs"] == [7, 6, 5, 8]
    assert a0 == [1] and a1 == [0]
    assert out[0]["relevant_chunks"] == []


def test_build_docceil30_giu_doc_vi30():
    pool = {"1": [[5, 0, 1.0]]}
    fhyde = {}
    out, a0, _ = build_docceil30(_sub(docs=[6, 8, 5]), pool, fhyde)
    assert out[0]["relevant_docs"] == [5, 6, 8]
    assert set(map(str, [6, 8, 5])) <= set(map(str, out[0]["relevant_docs"]))
    assert a0 == [2]


def test_build_docceil30_giu_doc_chunk_va_chunk_y_nguyen():
    chunks = [{"doc_id": 9, "chunk_text": "x"}, {"doc_id": 5, "chunk_text": "y"}]
    pool = {"1": [[5, 0, 1.0]]}
    fhyde = {}
    out, _, a1 = build_docceil30(_sub(chunks=chunks, docs=[5]), pool, fhyde)
    assert out[0]["relevant_chunks"] is chunks
    assert out[0]["relevant_docs"] == [5, 9]
    assert a1 == [1]


def test_cut_chunks_cat_cuoi_giu_docs():
    chunks = [{"doc_id": 1, "chunk_text": t} for t in "abcde"]
    out = build_docceil30(_sub(chunks=chunks, docs=[1]), {"1": []}, {})[0]
    cut = cut_chunks(out, 2)
    assert [c["chunk_text"] for c in cut[0]["relevant_chunks"]] == ["a", "b"]
    assert cut[0]["relevant_docs"] == out[0]["relevant_docs"]
    assert cut[0]["id"] == 1
    assert [c["chunk_text"] for c in cut_chunks(out, 0)[0]["relevant_chunks"]] == []


def test_zip_lzma_duoi_tran(tmp_path):
    p = tmp_path / "submission_docceil30.json"
    p.write_text(json.dumps(_sub(docs=[1, 2])), encoding="utf-8")
    z = tmp_path / "submission_docceil30.zip"
    assert zip_lzma(p, z) == z.stat().st_size > 0
