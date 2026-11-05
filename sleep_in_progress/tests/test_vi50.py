"""Fixed vi50 rules, missing-source handling, immutability and budget gate."""
import copy
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vi50 import forecast, fused, protect, target_docs


def fixture_data():
    return dict(pool={'1':[[1,0,3.],[2,1,2.],[3,2,1.]]},
                q8b={'1':[[2,1,4.],[1,0,3.]]},
                dense={'1':[[3,2,4.],[1,0,3.],[2,1,2.]]},
                hyde={'1':[[2,1,4.],[3,2,3.]]})


def test_sixth_rrf_source_exact():
    from fuse3 import fuse3
    data=fixture_data();bge={'1':{'1':1.,'3':4.}}
    d8={'1':{'2':9.,'1':2.}}
    actual=fused(data,bge,d8)
    expected=fuse3(data['pool'],data['q8b'],data['dense'],(1.,2.,2.,1.),60,data['hyde'],
                   [({'1':[[3,2,4.],[1,0,1.]]},1.),({'1':[[2,1,9.],[1,0,2.]]},1.)])
    assert actual==expected


def test_no_d8_is_original_five_sources():
    data=fixture_data()
    assert fused(data,{'1':{'1':2.}})==fused(data,{'1':{'1':2.}}, {})


def test_missing_d8_has_no_contribution_not_last_rank():
    data=fixture_data()
    before=fused(data,{})['1']
    after=fused(data,{}, {'1':{'2':1.}})['1']
    a={r[0]:r[2] for r in before};b={r[0]:r[2] for r in after}
    assert a[1]==b[1] and a[3]==b[3]
    assert b[2]==pytest.approx(a[2]+1/61,abs=2e-8)


def test_d8_ties_keep_pool_order():
    from vi37 import docwin_hits
    data=fixture_data()
    assert [h[0] for h in docwin_hits(data['pool'],{'1':{'1':2.,'2':2.,'3':2.}})['1']]==[1,2,3]


def test_target_n_plus_100_inclusive():
    rows={'1':[[i,i,1.] for i in range(300)]}
    assert target_docs(rows,{'1':60})=={'1':list(range(160))}


def test_target_short_pool():
    assert target_docs({'1':[[3,0,1.]]},{'1':100})=={'1':[3]}


def test_protection_exact_vi41c_rule():
    base=dict(id=1,relevant_docs=[1,2,3,4],relevant_chunks=[dict(doc_id=1,chunk_text='a'),dict(doc_id=3,chunk_text='b')])
    original=copy.deepcopy(base)
    row,n=protect(base,[[2,0,1.],[5,0,.9],[3,0,.8],[6,0,.7],[1,0,.6]])
    assert row['relevant_docs']==[2,5,3,1] and n==1
    assert row['relevant_chunks']==base['relevant_chunks'] and base==original


def test_protect_all_chunks_not_only_original_vi37():
    base=dict(id=1,relevant_docs=[1,2],relevant_chunks=[dict(doc_id=1,chunk_text='x'),dict(doc_id=2,chunk_text='y')])
    row,n=protect(base,[[3,0,1.],[4,0,.9],[1,0,.8],[2,0,.7]])
    assert row['relevant_docs']==[1,2] and n==2


def test_duplicate_candidate_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        protect(dict(id=1,relevant_docs=[1,2],relevant_chunks=[]),[[3,0,1.],[3,0,.9]])


def test_forecast_fixed_no_scan():
    f=forecast(264947)
    assert f['usd_upper']==pytest.approx(12.178360742525731)
    assert forecast(0)['usd_upper']>0
    assert forecast(2000000)['usd_upper']>40


def test_real_1200_build_checks_once(tmp_path,monkeypatch):
    import json
    import zipfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    import vi50 as v
    from vi37 import writez
    monkeypatch.setattr(v,'TEMP',tmp_path)
    source=tmp_path/'vi45';source.mkdir()
    (source/'bad_ids.json').write_text('[]')
    (source/'new_ids.json').write_text('[]')
    pq.write_table(pa.table({'id':[1,2,3]}),source/'links_corpus_vi_en_zh.parquet')
    refdir=tmp_path/'vi48';refdir.mkdir()
    base=[dict(id=q,relevant_docs=[1,2],relevant_chunks=[dict(doc_id=1,chunk_text='sample original medical text')]) for q in range(1,1201)]
    with zipfile.ZipFile(refdir/'submission_vi48.zip','w',compression=zipfile.ZIP_LZMA) as z:
        z.writestr('submission.json',json.dumps(base))
    out=tmp_path/'vi50';out.mkdir();(out/'outputs').mkdir()
    ctx=dict(data={s:{} for s in ('pool','q8b','dense','hyde')},bge={},rankings={},counts={},targets={})
    scores=[];expected=[]
    for q in range(1,1201):
        sq=str(q)
        for s in ctx['data']:
            ctx['data'][s][sq]=[[1,0,3.],[2,1,2.],[3,2,1.]]
        ctx['bge'][sq]={'1':3.,'2':2.,'3':1.}
        ctx['counts'][sq]=2;ctx['targets'][sq]=[1,2,3]
        for d in (1,2,3):
            c=v.cid(q,dict(doc_id=d,chunk_text=f'text-{d}'))
            r=dict(qid=sq,doc_id=d,cid=c,logit8c=float(d))
            scores.append(r);expected.append({k:r[k] for k in ('qid','doc_id','cid')})
    ctx['rankings']=v.fused(ctx['data'],ctx['bge'])
    writez(out/'fusion_context.json.zst',ctx)
    writez(out/'expected.json.zst',expected)
    writez(out/'reused_logits.json.zst',[])
    (out/'outputs/part.logits.json').write_text(json.dumps(scores))
    v.write(out/'prepare_report.json',dict(pairs=3600,new_pairs=3600,reused_pairs=0,top8=False))
    v.build(out)
    report=v.read(out/'build_report.json')
    assert report['validate']==report['check_zip']==0
    assert report['chunks_equal_vi48']==report['doc_counts_equal_vi48']==1200
    assert report['build_attempts']==1
    with pytest.raises(AssertionError,match='ONE build'):
        v.build(out)
