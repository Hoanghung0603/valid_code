import pandas as pd
import pytest

from scripts.prepare_private_champion import validate_queries, validate_row


def test_replay_requires_exact_text_not_just_ids():
    queries = pd.DataFrame({'id': [1, 2], 'query': ['original one', 'original two']})
    cached = pd.DataFrame({'query_id': [1, 1, 2], 'query_text': ['original one', 'original one', 'original two']})
    assert validate_queries(queries, cached, 2) == [1, 2]
    queries.loc[1, 'query'] = 'changed private question'
    with pytest.raises(ValueError, match='fresh retrieval'):
        validate_queries(queries, cached, 2)


def test_replay_rejects_conflicting_cache_and_duplicate_queries():
    q = pd.DataFrame({'id': [1, 2], 'query': ['a', 'b']})
    c = pd.DataFrame({'query_id': [1, 1], 'query_text': ['a', 'different']})
    with pytest.raises(ValueError, match='coverage'):
        validate_queries(q, c, 2)
    q.loc[1, 'id'] = 1
    with pytest.raises(ValueError, match='duplicate'):
        validate_queries(q, c, 2)


def test_schema_and_corpus_checks():
    row = {'id': 1, 'relevant_docs': [4], 'relevant_chunks': [{'doc_id': 4, 'chunk_text': 'source text'}]}
    assert validate_row(row, {4}) == (1, 1)
    with pytest.raises(ValueError, match='corpus'):
        validate_row(row, {5})
    row['relevant_chunks'].append(row['relevant_chunks'][0].copy())
    with pytest.raises(ValueError, match='Duplicate chunk'):
        validate_row(row, {4})
