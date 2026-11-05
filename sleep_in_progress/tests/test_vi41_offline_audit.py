import importlib.util
from pathlib import Path
import pytest
from judge_calib import metrics

spec=importlib.util.spec_from_file_location('vi41_offline_audit',Path(__file__).resolve().parents[1]/'deploy/vi41_offline_audit.py')
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def test_proxy_matches_fixed_calibration():
    rows=[{'id':1,'relevant_docs':[1,2,3]},{'id':2,'relevant_docs':[4,5]}]
    scores={'1':{1:2.,2:-1.},'2':{4:0.,5:3.}}
    got=audit.proxy(rows,scores)
    expected=metrics(rows,{q:{str(d):s for d,s in ds.items()} for q,ds in scores.items()},0.)
    assert got==pytest.approx(expected)
    assert got['missing']==1


def test_proxy_abstains_on_query_without_scores():
    got=audit.proxy([{'id':1,'relevant_docs':[1]}],{})
    assert got['r_proxy'] is None
    assert got['p_lower']==0 and got['p_upper']==1


def test_summary_top30_final_and_covered_docs():
    old=[{'id':q,'relevant_docs':[1],'relevant_chunks':[{'doc_id':1,'chunk_text':'a'}]} for q in range(1,1201)]
    new=[{'id':q,'relevant_docs':[1,2],
          'relevant_chunks':[{'doc_id':2,'chunk_text':'a'},{'doc_id':2,'chunk_text':'b'}]} for q in range(1,1201)]
    fresh={str(q):[2,3] for q in range(1,1201)}
    fused={str(q):[[2,0,.9],[1,0,.8],[3,0,.7]] for q in range(1,1201)}
    old8={str(q):{1:1.} for q in range(1,1201)}
    new8={str(q):{1:1.,2:-1.} for q in range(1,1201)}
    got=audit.summarize(old,new,fresh,fused,old8,new8)
    assert got['new_pool_docs_in_top30_per_query']==2
    assert got['new_pool_docs_in_doclist_per_query']==1
    assert got['new']['chunks_per_query']==2
    assert got['new']['docs_with_chunks_per_query']==1
    assert got['proxy_8b']['r_proxy_delta']==0
