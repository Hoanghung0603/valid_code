"""Bulk contract without GPU/network: exact identities, finite scores, durable parts."""
import hashlib
import json
from pathlib import Path
import sys
import types

import pytest
import zstandard as zstd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vi45


def setup_bulk(tmp_path,monkeypatch,badsha=False,badcount=False,nan=False):
    src,dest=tmp_path/'in',tmp_path/'out'
    src.mkdir()
    rows=[dict(qid='1',cid='a',doc_id=1,query='q',text='t'),
          dict(qid='1',cid='b',doc_id=2,query='q',text='u')]
    part=src/'pairs.part000.jsonl.zst'
    with zstd.open(part,'wt',encoding='utf-8') as f:
        f.write('\n'.join(map(json.dumps,rows)))
    config=dict(parts=[dict(name=part.name,count=3 if badcount else 2,
        sha256='bad' if badsha else hashlib.sha256(part.read_bytes()).hexdigest())])
    (src/'job.json').write_text(json.dumps(config))
    class Tok:
        def convert_tokens_to_ids(self,t):
            return {'yes':10,'no':11}[t]
    class LLM:
        def __init__(self,**kw):
            assert kw['max_model_len']==1025
            assert kw['max_logprobs']==2
        def get_tokenizer(self):
            return Tok()
        def collective_rpc(self,name):
            assert name=='install_yes_no'
            return [[10,11]]
        def generate(self,rows,params,**kwargs):
            return [types.SimpleNamespace(outputs=[types.SimpleNamespace(logprobs=[{
                10:types.SimpleNamespace(logprob=float('nan') if nan else -.1),
                11:types.SimpleNamespace(logprob=-2.1)}])]) for _ in rows]
    monkeypatch.setitem(sys.modules,'vllm',types.SimpleNamespace(__version__='0.15.1',LLM=LLM,
        SamplingParams=lambda **kwargs:kwargs))
    import qwen_rerank_run
    monkeypatch.setattr(qwen_rerank_run,'build_input_ids',lambda *args:[1,2,3])
    commits=[]
    monkeypatch.setitem(sys.modules,'modal',types.SimpleNamespace(Volume=types.SimpleNamespace(
        from_name=lambda name:types.SimpleNamespace(commit=lambda:commits.append(name)))))
    return src,dest,commits


def test_bulk_exact_scores_and_part_commit(tmp_path,monkeypatch):
    src,dest,commits=setup_bulk(tmp_path,monkeypatch)
    report=vi45.gpu('bulk',src,dest)
    scores=json.loads((dest/'pairs.part000.logits.json').read_text())
    assert [r['cid'] for r in scores]==['a','b']
    assert [r['logit8c'] for r in scores]==[2.,2.]
    assert report['parts'][0]['pairs']==2
    assert report['parts'][0]['tokens']==6
    assert commits==['r2ai-runner']
    with pytest.raises(AssertionError,match='duplicate'):
        vi45.gpu('bulk',src,dest)


@pytest.mark.parametrize('badsha,badcount,nan',[(True,False,False),(False,True,False),(False,False,True)])
def test_bulk_rejects_bad_input_or_score(tmp_path,monkeypatch,badsha,badcount,nan):
    src,dest,commits=setup_bulk(tmp_path,monkeypatch,badsha,badcount,nan)
    with pytest.raises(AssertionError):
        vi45.gpu('bulk',src,dest)
    assert not list(dest.glob('*.logits.json'))
    assert not commits
