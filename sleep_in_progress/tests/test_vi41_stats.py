import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vi41


def test_stats_new_docs_and_same_rank_logits(tmp_path,monkeypatch):
    monkeypatch.setattr(vi41,'ROOT',tmp_path)
    monkeypatch.setattr(vi41,'OUT',tmp_path/'out')
    old=[{'id':q,'relevant_docs':[1,2],'relevant_chunks':[]} for q in range(1,1201)]
    new=[{'id':q,'relevant_docs':[1,3],'relevant_chunks':[]} for q in range(1,1201)]
    vi41.write(tmp_path/'data/sub/vi37/submission_vi37.json',old)
    vi41.write(vi41.OUT/'submission_vi41.json',new)
    vi41.write(vi41.OUT/'new_ids.json',{str(q):[3] for q in range(1,1201)})
    vi41.write(vi41.OUT/'new_provenance.json',{str(q):{'3':['dense','bm25']} for q in range(1,1201)})
    oldscore={str(q):[[1,0,2.],[2,1,1.]] for q in range(1,1201)}
    newscore={str(q):[[1,0,2.],[3,2,3.]] for q in range(1,1201)}
    vi41.write(tmp_path/'data/sub/vi37/q8b_v37.json',oldscore)
    vi41.write(tmp_path/'data/sub/vi36/in1/pool_v36.json',oldscore)
    for key in ('q8b','pool'):
        vi41.write(vi41.OUT/'in1'/(key+'_v41.json'),newscore)
    vi41.stats()
    r=vi41.read(vi41.OUT/'stats_report.json')
    assert r['docs_in_per_query']==r['docs_out_per_query']==r['new_pool_docs_in_final_per_query']==1
    assert abs(r['jaccard_mean']-1/3)<1e-12
    assert r['new_final_by_source_per_query']=={'dense':1,'dchunk':0,'bm25':1,'neardup':0}
    assert r['matched_rank_logits']['8b']['new']['mean']==3
    assert r['matched_rank_logits']['8b']['vi37_same_rank']['mean']==1
    assert r['matched_rank_missing']['8b']=={'new':0,'vi37_same_rank':0}
