import sqlite3
from pathlib import Path

from src.retrieval.concept_graph import ConceptLexicon, add_document, create_graph, retrieve


LEXICON = Path(__file__).resolve().parents[1] / "configs/medical_concepts_v1.json"


def test_synonym_graph_is_evidence_bearing_and_deduplicated(tmp_path):
    terms = ConceptLexicon(LEXICON)
    db = create_graph(tmp_path / "graph.sqlite", terms)
    assert add_document(db, terms, 7, "https://example.org/7", "Phụ nữ mang thai có tăng huyết áp.") == 2
    assert add_document(db, terms, 9, "https://example.org/9", "Diabetes mellitus during pregnancy.") == 2
    assert add_document(db, terms, 12, "https://example.org/12", "Mang thai " + "khác " * 300 + "tăng huyết áp") == 2
    assert add_document(db, terms, 11, "https://example.org/11", "Không liên quan.") == 0
    db.commit()
    result = retrieve(db, terms, "Bà bầu bị cao huyết áp thì sao?")
    assert result[0]["doc_id"] == 7
    assert result[0]["overlap"] == 2
    assert result[0]["first_offset"] >= 0
    assert result[0]["url"] == "https://example.org/7"
    assert 12 not in [row["doc_id"] for row in result if row["overlap"] == 2]
    assert retrieve(db, terms, "triệu chứng không có trong từ điển") == []
    assert db.execute("SELECT COUNT(*) FROM mentions").fetchone()[0] == 6
    db.close()


def test_graph_refuses_overwrite(tmp_path):
    terms = ConceptLexicon(LEXICON)
    path = tmp_path / "graph.sqlite"
    create_graph(path, terms).close()
    try:
        create_graph(path, terms)
    except FileExistsError:
        pass
    else:
        raise AssertionError("Graph overwrite must be refused")
