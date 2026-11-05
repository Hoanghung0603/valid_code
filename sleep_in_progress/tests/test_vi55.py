import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vi55


def test_query_exact():
    assert vi55.query_text('đau?')=='Instruct: Given a Vietnamese medical question, retrieve web documents that answer the question\nQuery:đau?'
    assert vi55.REVISION=='1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'


def test_gate_and_cost():
    assert vi55.repeat_pass(.999,.9999)
    assert not vi55.repeat_pass(.99899,1.)
    assert not vi55.repeat_pass(1.,.99989)
    r=vi55.forecast(1000000,10000,jobs=2,overhead=10)
    assert r['gpu_seconds']==150 and r['usd']==150*vi55.RATE


def test_batches_padded_budget():
    rows=[dict(input_ids=list(range(n))) for n in (9,3,4,8,2)]
    batches=list(vi55.token_batches(rows,16))
    assert sorted(i for b in batches for i in b)==list(range(5))
    assert all(max(len(rows[i]['input_ids']) for i in b)*len(b)<=16 for b in batches)


def test_topk_ties_and_merge():
    s=np.array([[.8,.7]],dtype=np.float32);ids=np.array([[5,2]])
    ss,ii=vi55.merge_topk(s,ids,np.array([[.8,.9]],dtype=np.float32),np.array([[1,9]]),3)
    assert ii.tolist()==[[9,1,5]]
    assert np.allclose(ss,[[.9,.8,.8]])


def test_metrics_query_pool_not_global_only():
    top={'1':[[d,1/d] for d in range(1,301)],'2':[[d,1/d] for d in range(301,601)]}
    pool={'1':[1,2,3,301],'2':[301,302,1]}
    dense={'1':[[d,0,0] for d in range(1,201)],'2':[[d,0,0] for d in range(301,501)]}
    docs={'1':[1,100,250],'2':[301,450,550]}
    exact={'1':{'1':1,'100':100,'250':250},'2':{'301':1,'450':150,'550':250}}
    m=vi55.metrics(top,pool,dense,docs,exact)
    assert m['100']['outside_query_pool']['mean']==97.5
    assert m['100']['vi53_coverage_micro']==.5
    assert m['200']['vi53_coverage_micro']==4/6
    assert m['vi53_rank']['exact'] and m['vi53_rank']['missing_or_beyond1000']==0
    assert m['top30_proxy']['outside_global_pool_fraction']<m['top30_proxy']['new_fraction']+1e-9
    assert sum(m['top30_cosine'][k]['n'] for k in ('new_to_query_pool','old_in_query_pool'))==60
    assert m['top30_cosine']['old_in_query_pool']['n']==5


def test_topk_chunked_equals_exhaustive():
    rng=np.random.default_rng(55)
    q=rng.normal(size=(3,4)).astype(np.float32)
    docs=rng.normal(size=(91,4)).astype(np.float32)
    scores=q@docs.T
    best=np.empty((3,0),np.float32);ids=np.empty((3,0),np.int64)
    for start in range(0,91,13):
        ss=scores[:,start:start+13]
        ii=np.broadcast_to(np.arange(start,start+ss.shape[1]),ss.shape)
        best,ids=vi55.merge_topk(best,ids,ss,ii,10)
    order=np.argsort(-scores,axis=1)[:,:10]
    assert np.array_equal(ids,order)
    assert np.allclose(best,np.take_along_axis(scores,order,axis=1))


def test_token_budget_single_max_and_empty():
    assert list(vi55.token_batches([]))==[]
    rows=[dict(input_ids=[1]*1024) for _ in range(33)]
    bs=list(vi55.token_batches(rows))
    assert list(map(len,bs))==[16,16,1]


def test_partition_top1000_sufficient_and_rank_additive():
    rng=np.random.default_rng(555)
    values=np.round(rng.normal(size=5000),6)
    wanted=[17,52,1999,3000]
    left=np.arange(0,2500);right=np.arange(2500,5000)
    partial=[]
    for ids in (left,right):
        partial.extend(sorted(ids,key=lambda i:(-values[i],i))[:1000])
    merged=sorted(partial,key=lambda i:(-values[i],i))[:1000]
    exhaustive=sorted(range(5000),key=lambda i:(-values[i],i))[:1000]
    assert merged==exhaustive
    for d in wanted:
        combined=1+sum(int(np.sum(values[ids]>values[d])) for ids in (left,right))
        assert combined==1+int(np.sum(values>values[d]))


def test_rank_sorted_search_with_ties():
    # Modal torch.searchsorted(right=True) has the same competition-rank rule.
    values=np.array([[.4,.2,.4,.7],[-.1,.1,.1,.2]])
    thresholds=np.array([[.4,.2],[-.1,.1]])
    got=np.asarray([[len(row)-np.searchsorted(np.sort(row),th,side='right') for th in ts]
                    for row,ts in zip(values,thresholds)])
    assert got.tolist()==[[1,3],[3,1]]
    assert np.array_equal(got,(values[:,:,None]>thresholds[:,None,:]).sum(axis=1))


def test_fp16_renormalization_cosine():
    rng=np.random.default_rng(551)
    vectors=rng.normal(size=(17,4096)).astype(np.float32)
    vectors/=np.linalg.norm(vectors,axis=1)[:,None]
    saved=vectors.astype(np.float16)
    norm=saved.astype(np.float32);norm/=np.linalg.norm(norm,axis=1)[:,None]
    assert np.max(np.abs(norm@norm.T-vectors@vectors.T))<1e-4


def test_preempt_partial_rerun_equals_clean(tmp_path):
    import json
    import pytest
    import zstandard as zstd
    source=tmp_path/'in';source.mkdir();part='docs55.part016.jsonl.zst'
    with zstd.open(source/part,'wt',encoding='utf-8') as f:
        for d in range(7):f.write(json.dumps(dict(id=d,input_ids=[d+1]))+'\n')
    clean=tmp_path/'clean';resumed=tmp_path/'resumed';resumed.mkdir()
    def embed(rows):
        v=np.asarray([[r['id']+1,1,2,3] for r in rows],np.float32)
        return v/np.linalg.norm(v,axis=1)[:,None]
    def preempt(rows):
        np.savez(resumed/(part+'.emb.npz'),ids=np.array([0]),emb=embed(rows[:1]).astype(np.float16))
        raise RuntimeError('simulated preemption')
    with pytest.raises(RuntimeError):vi55.process_part(source,resumed,part,preempt,dimension=4)
    assert vi55.committed_part(source,resumed,part,4) is None
    commits=[]
    r=vi55.process_part(source,resumed,part,embed,dimension=4,commit=lambda:commits.append(1))
    vi55.process_part(source,clean,part,embed,dimension=4)
    assert r['attempt']==2 and len(commits)==2
    assert vi55.process_part(source,resumed,part,lambda _:pytest.fail('must skip'),dimension=4)['skipped']
    for k in ('ids','emb'):
        assert np.array_equal(np.load(resumed/(part+'.emb.npz'))[k],np.load(clean/(part+'.emb.npz'))[k])
    z=np.load(resumed/(part+'.emb.npz'));np.savez(resumed/(part+'.emb.npz'),ids=z['ids'],emb=np.full((7,4),np.nan,dtype=np.float16))
    assert vi55.committed_part(source,resumed,part,4) is None
    with pytest.raises(AssertionError,match='twice'):vi55.process_part(source,resumed,part,embed,dimension=4)


def test_embedding_audit_rejects_wrong_ids_shape_norm_and_sha(tmp_path):
    import pytest
    p=tmp_path/'x.npz'
    for ids,emb in [([1],[[1,0]]),([1,2],[[1,0]]),([1,2],[[1,0],[0,2]]),([1,1],[[1,0],[0,1]])]:
        np.savez(p,ids=ids,emb=np.asarray(emb,np.float16))
        with pytest.raises(AssertionError):vi55.audit_embedding(p,[1,2],2)
    np.savez(p,ids=[1,2],emb=np.eye(2,dtype=np.float16))
    assert vi55.audit_embedding(p,[1,2],2)['documents']==2


def test_part_requires_completion_marker_input_sha_and_output_sha(tmp_path):
    import json
    import zstandard as zstd
    source=tmp_path/'in';source.mkdir();dest=tmp_path/'out';dest.mkdir();part='p.jsonl.zst'
    def save_source():
        with zstd.open(source/part,'wt',encoding='utf-8') as f:
            for d in [1,2]:f.write(json.dumps(dict(id=d,input_ids=[d]))+'\n')
    save_source()
    def embed(_):return np.eye(2,dtype=np.float32)
    np.savez(dest/(part+'.emb.npz'),ids=[1,2],emb=np.eye(2,dtype=np.float16))
    assert vi55.committed_part(source,dest,part,2) is None
    vi55.process_part(source,dest,part,embed,dimension=2)
    assert vi55.committed_part(source,dest,part,2) is not None
    marker=dest/(part+'.complete.json');report=vi55.read(marker)
    report['output']['sha256']='0'*64;vi55.write(marker,report)
    assert vi55.committed_part(source,dest,part,2) is None
    (dest/(part+'.emb.npz')).write_bytes(b'broken npz')
    assert vi55.committed_part(source,dest,part,2) is None
    vi55.process_part(source,dest,part,embed,dimension=2)
    assert vi55.committed_part(source,dest,part,2) is not None
    with (source/part).open('ab') as f:f.write(b'\0')
    assert vi55.committed_part(source,dest,part,2) is None
