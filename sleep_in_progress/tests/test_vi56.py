import json
from pathlib import Path
import sys

import pytest
import zstandard as zstd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vi56


def part(path,rows):
    with zstd.open(path,'wt',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r)+'\n')


def test_select_fixed_top100_no_search():
    top={'1':[[d,1.] for d in range(1,202)]}
    assert vi56.select(top,{'1':[[1,0,0.],[7,0,0.]]},set(range(1,202)))=={'1':[d for d in range(1,101) if d not in (1,7)]}


def test_select_allowed_only():
    assert vi56.select({'1':[[d,0.] for d in range(100)]},{'1':[]},{5,6})=={'1':[5,6]}


def test_forecast_cap():
    assert vi56.forecast(100000,300000)['usd_upper']<10
    assert vi56.forecast(400000,300000)['usd_upper']>10


def test_complete_requires_exact_ids_sha(tmp_path):
    source=tmp_path/'x.jsonl.zst'; dest=tmp_path/'out.json'
    rows=[dict(identity=str(i),qid='1',doc_id=i,idx=i,kind='d8w',text='x',query='q') for i in range(4)]
    part(source,rows)
    vi56.process_part(source,dest,lambda rs:[float(i) for i in range(len(rs))])
    assert vi56.completed_part(source,dest)
    assert vi56.process_part(source,dest,lambda rs:pytest.fail('must not rescore'))['skipped']
    got=vi56.read(dest);got.pop();vi56.write(dest,got)
    assert vi56.completed_part(source,dest) is None


def test_preempted_part_rerun_equal_clean(tmp_path):
    source=tmp_path/'x.jsonl.zst'
    rows=[dict(identity=str(i),qid='1',doc_id=i,idx=i,kind='d8w',text='x',query='q') for i in range(5)]
    part(source,rows); resumed=tmp_path/'resumed.json';clean=tmp_path/'clean.json'
    def fail(rs): raise RuntimeError('preempted')
    with pytest.raises(RuntimeError):vi56.process_part(source,resumed,fail)
    vi56.process_part(source,resumed,lambda rs:[2.]*len(rs))
    vi56.process_part(source,clean,lambda rs:[2.]*len(rs))
    assert vi56.read(resumed)==vi56.read(clean)


def test_preemption_max_one_rerun(tmp_path):
    source=tmp_path/'x.jsonl.zst';part(source,[dict(identity='a',qid='1',doc_id=1,idx=0,kind='d8w')])
    def fail(rs):raise RuntimeError('preempted')
    for _ in range(2):
        with pytest.raises(RuntimeError):vi56.process_part(source,tmp_path/'out.json',fail)
    with pytest.raises(AssertionError,match='twice'):vi56.process_part(source,tmp_path/'out.json',fail)


def test_extend_exact_vi41_last_rank_fallback():
    ctx={'data':{s:{'1':[[1,0,1.]]} for s in ('pool','q8b','dense','hyde')}}
    fresh={'1':[2]};scores={'1':{'2':dict(bge_doc=.5,q8b_doc=2.,bge_win=3.,d8w=4.,q3e=.7)}}
    data,bge,q3e,d8w=vi56.extend(ctx,{'1':{'1':1.}},{'1':{'1':.8}},{'1':{'1':1.}},fresh,scores,{'2':{'idx':1}})
    assert data['dense']['1']==[[1,0,1.],[2,1,-1.]]
    assert data['hyde']['1']==data['dense']['1']
    assert [r[0] for r in data['q8b']['1']]==[2,1]
    assert ctx['data']['pool']['1']==[[1,0,1.]]
    assert bge['1']['2']==3. and q3e['1']['2']==.7 and d8w['1']['2']==4.


def test_empty_missing_source_uses_original_no_contribution():
    ctx={'data':{s:{'1':[[1,0,1.]]} for s in ('pool','q8b','dense','hyde')}}
    data,bge,q3e,d8w=vi56.extend(ctx,{}, {'1':{'1':1.}}, {},{'1':[2]},{'1':{'2':{'q3e':.5}}},{'2':{'idx':1}})
    assert data['pool']['1'][-1][2]==-10000.
    assert len(data['q8b']['1'])==1 and not bge and not d8w


@pytest.mark.parametrize('kind', ['qwen', 'bge'])
def test_gpu_cli_sdk_free_bench_reads_real_part(tmp_path, monkeypatch, kind):
    import builtins
    original_import = builtins.__import__
    def no_modal(name, *args, **kwargs):
        if name == 'modal':
            raise ModuleNotFoundError("No module named 'modal'")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', no_modal)
    def fake_loader(stage, metrics):
        assert stage == kind
        def score(rows):
            assert all(r['query'] == 'q' and r['text'] == 'body' for r in rows)
            metrics['tokens'] += 5*len(rows)
            return [float(r['doc_id']) for r in rows]
        return score
    monkeypatch.setattr(vi56, 'load_scorer', fake_loader)
    src=tmp_path/'in';src.mkdir();out=tmp_path/'out'
    name=kind+'.part000.jsonl.zst'
    part(src/name,[dict(identity=str(i),qid='1',doc_id=i,idx=i,kind='d8w',query='q',text='body') for i in range(5)])
    vi56.main(['gpu','--kind',kind,'--job','bench_'+kind,'--src',str(src),'--out',str(out),
               '--parts',name,'--bench'])
    report=vi56.read(out/'report.json')
    assert report['pairs']==10 and report['tokens']==50 and report['pearson']==1.
    assert report['pairs_per_second']>0


def test_gpu_checks_paths_before_model_load(tmp_path, monkeypatch):
    monkeypatch.setattr(vi56, 'load_scorer', lambda *args: pytest.fail('no model load'))
    with pytest.raises(AssertionError):
        vi56.gpu('qwen','missing',['absent.zst'],10,src=tmp_path,dest=tmp_path/'out')


def test_aggregate_complete_superset_filters_excluded_and_maxes_windows(tmp_path):
    fresh={'1':[1]}
    vi56.write(tmp_path/'new_ids.json',fresh)
    vi56.write(tmp_path/'q3e_new.json',{'1':{'1':.8}})
    vi56.write(tmp_path/'prepare_report.json',{'empty_pairs':[['1',3]]})
    inputs=tmp_path/'modal_input';inputs.mkdir()
    out=tmp_path/'outputs'/'worker';out.mkdir(parents=True)
    for stage,kinds in [('bge',['bge_doc','bge_win','bge_win']),('qwen',['q8b_doc','d8w','d8w'])]:
        rows=[dict(identity=f'{stage}:{d}:{i}',qid='1',doc_id=d,idx=d,kind=kind)
              for d in (1,2) for i,kind in enumerate(kinds)]
        name=stage+'.part000.jsonl.zst';part(inputs/name,rows)
        vi56.write(tmp_path/f'expected_{stage}.json',[[r[k] for k in ('identity','qid','doc_id','kind')] for r in rows])
        vi56.process_part(inputs/name,out/(name+'.scores.json'),lambda rs:[float(i) for i in range(len(rs))])
    scores=vi56.aggregate(tmp_path)
    assert set(scores['1'])=={'1'}
    assert scores['1']['1']==dict(q3e=.8,bge_doc=0.,bge_win=2.,q8b_doc=0.,d8w=2.)


def test_aggregate_rejects_missing_part(tmp_path):
    vi56.write(tmp_path/'new_ids.json',{'1':[1]})
    vi56.write(tmp_path/'q3e_new.json',{'1':{'1':.8}})
    vi56.write(tmp_path/'prepare_report.json',{'empty_pairs':[]})
    vi56.write(tmp_path/'expected_bge.json',[['missing','1',1,'bge_doc']])
    vi56.write(tmp_path/'expected_qwen.json',[])
    with pytest.raises(AssertionError,match='missing scores'):
        vi56.aggregate(tmp_path)


def test_extended_fusion_exact_fixed_seven_source_weights_and_protection():
    from vi52 import fused
    from vi50 import protect
    ctx={'data':{s:{'1':[[1,0,1.],[3,2,.5]]} for s in ('pool','q8b','dense','hyde')}}
    fresh={'1':[2]};scores={'1':{'2':dict(bge_doc=2.,q8b_doc=2.,bge_win=3.,d8w=4.,q3e=.9)}}
    data,bge,q3e,d8w=vi56.extend(ctx,{'1':{'1':1.,'3':.5}},
        {'1':{'1':.8,'3':.7}},{'1':{'1':1.,'3':.5}},fresh,scores,{'2':{'idx':1}})
    rankings=fused(data,bge,d8w,q3e)
    # New doc ranks first in five scorers (weight 6), last in dense+HyDE (weight 3).
    assert next(h[2] for h in rankings['1'] if h[0]==2)==round(6/61+3/63,8)
    base=dict(id=1,relevant_docs=[1,3],relevant_chunks=[dict(doc_id=3,chunk_text='protected')])
    row,_=protect(base,rankings['1'])
    assert row['relevant_docs']==[2,3]
    assert row['relevant_chunks']==base['relevant_chunks']
