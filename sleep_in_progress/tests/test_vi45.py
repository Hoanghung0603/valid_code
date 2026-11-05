import math
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vi45


def row(d,t,s):
    c=dict(doc_id=d,chunk_text=t,s=s,zc=s,zd=0)
    c['cid']=vi45.cid('1',c)
    return c


def test_identity_doc_sensitive():
    assert vi45.cid(1,row(1,'a',0)) != vi45.cid(1,row(2,'a',0))


def test_zero_preserves_original_rank_with_bonus():
    rows=[row(1,'a',3),row(1,'b',2.5),row(2,'c',2)]
    assert [c['chunk_text'] for c in vi45.rank(rows)] == ['a','c','b']
    assert [vi45.key(c) for c in vi45.rank(rows,{c['cid']:0 for c in rows})] == [vi45.key(c) for c in vi45.rank(rows)]


def test_unscored_after_every_scored():
    rows=[row(1,'a',100),row(2,'b',-100)]
    assert vi45.rank(rows,{rows[1]['cid']:0})[0]['chunk_text']=='b'


def test_exact_k_policy():
    rows=[row(i,str(i),-i) for i in range(500)]
    assert len(vi45.shortlist(rows,12))==200
    assert len(vi45.shortlist(rows,120))==240


def test_nonfinite_rejected():
    import pytest
    c=row(1,'a',0)
    with pytest.raises(AssertionError):
        vi45.rank([c],{c['cid']:math.nan})


def test_z8_population_is_scored_valid_set():
    rows=[row(1,'a',0),row(2,'b',0),row(3,'c',100)]
    got=vi45.rank(rows,{rows[0]['cid']:1,rows[1]['cid']:3})
    assert got[0]['chunk_text']=='b'
    assert got[0]['z8c']==1
    assert got[1]['z8c']==-1
    assert got[2]['chunk_text']=='c'


def test_vi44_supplement_only_new_bridges_not_old_pairs():
    rows=[row(1,'old',2),row(2,'u900',0)]
    bridges=[dict(doc_id=1,chunk_text='old',score=7,zd=0),
             dict(doc_id=3,chunk_text='new51',score=1,zd=0)]
    got=vi45.supplement_rows('1',rows,bridges,2,0,1)
    assert [c['chunk_text'] for c in got]==['new51']
    assert len(rows)==3
    assert got[0]['s']==got[0]['zc']+4*got[0]['zd']


def test_vi44_supplement_respects_new_k():
    rows=[row(i,str(i),1000-i) for i in range(250)]
    got=vi45.supplement_rows('1',rows,[dict(doc_id=999,chunk_text='tail',score=-100,zd=0)],10,0,1)
    assert not got
    assert len(rows)==251


def test_assemble_preserves_docs_and_counts():
    rows=[row(1,'a',0),row(2,'b',0),row(3,'c',0)]
    ref=[dict(id=1,relevant_docs=[1,2,3],relevant_chunks=[dict(doc_id=1,chunk_text='a')])]
    sub,stats=vi45.assemble(ref,{'1':rows},{'1':{rows[2]['cid']:1}})
    assert sub[0]['relevant_docs']==ref[0]['relevant_docs']
    assert sub[0]['relevant_chunks']==[dict(doc_id=3,chunk_text='c')]
    assert stats==[dict(qid='1',jaccard=0.,replaced=1)]


def test_assemble_rejects_unknown_logit_and_duplicate_candidates():
    import pytest
    rows=[row(1,'a',0)]
    ref=[dict(id=1,relevant_docs=[1],relevant_chunks=[dict(doc_id=1,chunk_text='a')])]
    with pytest.raises(AssertionError):
        vi45.assemble(ref,{'1':rows},{'1':{'unknown':0}})
    with pytest.raises(AssertionError):
        vi45.assemble(ref,{'1':rows+rows},{'1':{rows[0]['cid']:0}})


def test_assemble_rejects_chunk_outside_doclist():
    import pytest
    rows=[row(2,'b',0)]
    ref=[dict(id=1,relevant_docs=[1],relevant_chunks=[dict(doc_id=1,chunk_text='a')])]
    with pytest.raises(AssertionError):
        vi45.assemble(ref,{'1':rows},{'1':{rows[0]['cid']:0}})


def test_all_scored_zero_matches_global_union_not_appended_prefix():
    rows=[row(1,'u900',0),row(2,'bridge51',2)]
    ref=[dict(id=1,relevant_docs=[1,2],relevant_chunks=[dict(doc_id=1,chunk_text='u900')])]
    sub,stats=vi45.assemble(ref,{'1':rows},{'1':{c['cid']:0 for c in rows}})
    assert sub[0]['relevant_chunks'][0]['chunk_text']=='bridge51'
    assert stats[0]['replaced']==1
