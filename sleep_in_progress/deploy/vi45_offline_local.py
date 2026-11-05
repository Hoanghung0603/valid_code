"""Independent final ranked-output and offline-stat recomputation, CPU only."""
import json
import math
from pathlib import Path
import statistics
import sys

from scipy.stats import spearmanr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import vi45
from dwmax_diag import read
from vi41c import load_zip
from chunk_docfuse import zscores

T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi45')
cache=read(T/'candidates_vi45_vi44.json.zst')
ref=load_zip(T.parent/'vi44/submission_vi44.zip')
actual={str(r['id']):r for r in load_zip(T/'submission_vi45.zip')}
logits={q:{} for q in cache}
for p in (T/'outputs').glob('*/*/*.logits.json'):
    for r in read(p):
        q=str(r['qid'])
        assert r['cid'] not in logits[q]
        logits[q][r['cid']]=r['logit8c']
expected,stats=vi45.assemble(ref,cache,logits)
assert all(r==actual[str(r['id'])] for r in expected)
a,b,rhos=[],[],[]
for q,rows in cache.items():
    pairs=[(logits[q][c['cid']],c['zc']) for c in rows if c['cid'] in logits[q]]
    x,y=zip(*pairs)
    a.extend(zscores(list(x)))
    b.extend(y)
    rho=float(spearmanr(x,y).statistic)
    if math.isfinite(rho):
        rhos.append(rho)
computed=dict(jaccard_macro=statistics.mean(s['jaccard'] for s in stats),
    replaced_total=sum(s['replaced'] for s in stats),changed_queries=sum(s['replaced']>0 for s in stats),
    spearman_z8c_zc_pooled=float(spearmanr(a,b).statistic),spearman_z8c_zc_macro=statistics.mean(rhos),
    spearman_finite_queries=len(rhos))
remote=read(T/'offline_vi45.json')
assert remote['per_query']==stats
for k,v in computed.items():
    assert abs(remote[k]-v)<1e-10,(k,remote[k],v)
per_query=[]
for s in stats:
    row=actual[s['qid']]
    per_query.append(dict(**s,chunk_bearing_docs=len({str(c['doc_id']) for c in row['relevant_chunks']}),
                          chunks=len(row['relevant_chunks'])))
result=dict(ranked_output_equal=1200,offline_equal=True,**computed,
    replaced_mean=computed['replaced_total']/1200,
    chunk_bearing_docs_mean=statistics.mean(s['chunk_bearing_docs'] for s in per_query),
    chunks_mean=statistics.mean(s['chunks'] for s in per_query),per_query=per_query)
(T/'local_offline_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k!='per_query'}))
