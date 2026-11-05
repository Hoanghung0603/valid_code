from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))

from vi54 import document_text, mean_z, query_text, require_raw


def test_prefix():
    assert query_text('câu hỏi') == 'query: câu hỏi'
    assert document_text('title', 'body') == 'passage: title\nbody'


def test_single_source_preserves_order():
    pool = {'1': [[10], [20], [30]]}
    source = {'1': {'10': .1, '20': .8, '30': .2}}
    fused = mean_z(pool, source, {})['1']
    assert sorted(fused, key=fused.get) == ['10', '30', '20']
    assert sum(fused.values()) == pytest.approx(0, abs=1e-12)


def test_mean_z_missing_and_scale_invariance():
    pool = {'1': [[10], [20], [30]]}
    a = {'1': {'10': 1, '20': 2, '30': 3}}
    b = {'1': {'10': 10, '20': 20}}
    merged = mean_z(pool, a, b)['1']
    assert merged['30'] == pytest.approx(mean_z(pool, a)['1']['30'])
    assert mean_z(pool, a) == mean_z(pool, {'1': {'10': 10, '20': 20, '30': 30}})


def test_reject_invalid_or_missing():
    with pytest.raises(AssertionError):
        mean_z({'1': [[10]]}, {})
    with pytest.raises(AssertionError):
        mean_z({'1': [[10]]}, {'1': {'10': float('nan')}})


def test_provenance_gate_rejects_qwen_tokens():
    with pytest.raises(ValueError, match='INPUT_PROVENANCE'):
        require_raw([{'id': 1, 'input_ids': [1, 2]}])
    assert require_raw([{'id': 1, 'title': '', 'text': 'clean'}])
    assert require_raw([{'id': 1, 'text': 'title\nclean'}])
    with pytest.raises(ValueError):
        require_raw([{'id': 1, 'text': ''}])


def test_nemotron_tokenization_not_qwen_ids():
    from vi54 import tokenize
    class Tok:
        def __call__(self, texts, truncation, max_length):
            self.texts=texts;self.limit=max_length
            return {'input_ids':[[3,4] for t in texts]}
    tok=Tok()
    assert tokenize(tok,[{'id':9,'text':'title\nbody'}])==[{'id':9,'input_ids':[3,4]}]
    assert tok.texts==['passage: title\nbody'] and tok.limit==2048
    tokenize(tok,[{'id':1,'text':'hỏi'}],True)
    assert tok.texts==['query: hỏi'] and tok.limit==256


def test_token_batches_padded_budget():
    from vi54 import token_batches
    rows=[dict(input_ids=[0]*n) for n in (3,7,2,5,1)]
    batches=list(token_batches(rows,10))
    assert sorted(i for b in batches for i in b)==list(range(5))
    assert all(max(len(rows[i]['input_ids']) for i in b)*len(b)<=10 for b in batches)


def test_encoder_sdpa_explicit_noncausal_and_offline():
    # 5.2 Ministral attention has a hardcoded causal default even for the encoder
    # checkpoint; the GPU implementation must pass the override on every forward.
    source=(Path(__file__).resolve().parents[1]/'scripts/vi54.py').read_text(encoding='utf-8')
    assert 'model(**batch,is_causal=False,use_cache=False)' in source
    assert "local_files_only=True" in source
    deploy=(Path(__file__).resolve().parents[1]/'deploy/modal_vi54.py').read_text(encoding='utf-8')
    assert '.run_commands(' not in deploy
    assert "HF_HUB_OFFLINE':'1'" in deploy


def test_meanz_population_expected():
    pool={'1':[[1],[2],[3]]}
    a={'1':{'1':1.,'2':2.,'3':3.}}
    b={'1':{'1':3.,'2':1.,'3':2.}}
    result=mean_z(pool,a,b)['1']
    assert result['1']==pytest.approx(0.)
    assert result['2']==pytest.approx(-.6123724356957945)
    assert result['3']==pytest.approx(.6123724356957945)


def test_one_dense_source_reuses_vi52():
    from test_vi52 import example
    import vi52
    from vi54 import fused
    data, bge, d8w, q3e = example()
    assert fused(data, bge, d8w, q3e, {}) == vi52.fused(data, bge, d8w, q3e)
