import copy
import json
from pathlib import Path
import sys
import zipfile

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vi51 as v


def test_prompt_exact():
    assert v.query_text('câu hỏi')=='Instruct: Given a Vietnamese medical question, retrieve web documents that answer the question\nQuery:câu hỏi'


def test_pins_and_release():
    assert all(len(rev)==40 for name,rev in v.MODELS.values())
    assert v.RELEASE<'2026-08-01'


def test_repeat_gate():
    assert v.repeat_pass(.999,.9999)
    assert not v.repeat_pass(.99899,1.)
    assert not v.repeat_pass(.9995,.99989)


def test_length_sorted_padded_token_batches_restore_order():
    rows=[dict(input_ids=[0]*n) for n in (7,2,8,3,4)]
    batches=list(v.token_batches(rows,16))
    assert sorted(i for b in batches for i in b)==list(range(5))
    assert all(max(len(rows[i]['input_ids']) for i in b)*len(b)<=16 for b in batches)
    assert [i for b in batches for i in b]==[1,3,4,0,2]


def test_full_pool_extra_rrf():
    from fuse3 import fuse3
    from vi37 import docwin_hits
    data={s:{'1':[[1,0,3.],[2,1,2.],[3,2,1.]]} for s in ('pool','q8b','dense','hyde')}
    bge={'1':{'1':2.}};new={'1':{'1':.1,'2':.2,'3':.3}}
    assert v.fused(data,bge,new)==fuse3(data['pool'],data['q8b'],data['dense'],(1.,2.,2.,1.),60,data['hyde'],[(docwin_hits(data['pool'],bge),1.),(docwin_hits(data['pool'],new),1.)])


def test_protection_vi48_chunks_count_immutable():
    base=dict(id=1,relevant_docs=[1,2,3,4],relevant_chunks=[dict(doc_id=1,chunk_text='x'),dict(doc_id=3,chunk_text='y')])
    old=copy.deepcopy(base)
    row,n=v.protect(base,[[2,0,1.],[5,1,.9],[3,2,.8],[6,3,.7],[1,4,.6]])
    assert row['relevant_docs']==[2,5,3,1] and n==1
    assert row['relevant_chunks']==base['relevant_chunks'] and old==base


def test_duplicate_rejected():
    with pytest.raises(ValueError,match='duplicate'):
        v.protect(dict(id=1,relevant_docs=[1,2],relevant_chunks=[]),[[3,0,1.],[3,1,.8]])


def test_real_1200_build_once(tmp_path,monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from vi37 import writez
    monkeypatch.setattr(v,'TEMP',tmp_path)
    source=tmp_path/'vi45';source.mkdir()
    (source/'bad_ids.json').write_text('[]');(source/'new_ids.json').write_text('[]')
    pq.write_table(pa.table({'id':[1,2,3]}),source/'links_corpus_vi_en_zh.parquet')
    ref=tmp_path/'vi48';ref.mkdir()
    base=[dict(id=q,relevant_docs=[1,2],relevant_chunks=[dict(doc_id=1,chunk_text='original medical text')]) for q in range(1,1201)]
    with zipfile.ZipFile(ref/'submission_vi48.zip','w',compression=zipfile.ZIP_LZMA) as z:z.writestr('submission.json',json.dumps(base))
    shared=tmp_path/'vi51';shared.mkdir();out=tmp_path/'vi51a';out.mkdir();(out/'outputs').mkdir()
    ctx=dict(data={s:{} for s in ('pool','q8b','dense','hyde')},bge={})
    scores={}
    for q in range(1,1201):
        sq=str(q)
        for s in ctx['data']:ctx['data'][s][sq]=[[1,0,3.],[2,1,2.],[3,2,1.]]
        ctx['bge'][sq]={'1':3.,'2':2.,'3':1.};scores[sq]={'1':.1,'2':.2,'3':.3}
    ctx['rankings']=v.fused(ctx['data'],ctx['bge'])
    writez(shared/'fusion_context.json.zst',ctx);v.write(shared/'prepare_report.json',dict(pairs=3600,documents=3))
    writez(out/'outputs/cosines.json.zst',scores)
    v.build(out,'a',shared)
    report=v.read(out/'build_report.json')
    assert report['validate']==report['check_zip']==0
    assert report['chunks_equal_vi48']==report['doc_counts_equal_vi48']==1200
    assert report['build_attempts']==1
    import vi51_audit
    monkeypatch.setattr(vi51_audit,'TEMP',tmp_path)
    vi51_audit.audit(out)
    assert v.read(out/'independent_audit.json')['chunks_within_docs']
    with pytest.raises(AssertionError,match='ONE build'):v.build(out,'a',shared)
