"""Independently reconstruct top1000/coverage from saved partition outputs."""
from pathlib import Path
import sys
import math
import statistics
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,fingerprint
T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi55')
r=read(T/'phase1_report.json');top=read(T/'q3e8b_full_top1000.json.zst')
assert fingerprint(T/'q3e8b_full_top1000.json.zst')==r['output']
a=read(T/'search_<MODAL_PROFILE_B>/top1000.json.zst');b=read(T/'search_<MODAL_PROFILE_C>/top1000.json.zst')
allowed=set(map(int,np.load(T/'all_ids.npy')));pool=read(T/'pool.json');docs=read(T/'vi53_doclists.json')
assert len(top)==1200 and set(map(int,top))==set(range(1,1201))
totals={100:0,200:0};novel={100:[],200:[]};global_pool={d for ds in pool.values() for d in ds}
for q,rows in top.items():
    assert rows==sorted(a[q]+b[q],key=lambda x:(-x[1],x[0]))[:1000]
    assert len(rows)==1000 and len({d for d,s in rows})==1000
    assert all(d in allowed and math.isfinite(s) for d,s in rows)
    for k in (100,200):
        ids={d for d,s in rows[:k]};totals[k]+=len(ids&set(docs[q]));novel[k].append(len(ids-global_pool))
for k in (100,200):
    assert totals[k]==r['metrics'][str(k)]['vi53_found']
    assert statistics.mean(novel[k])==r['metrics'][str(k)]['outside_global_pool']['mean']
assert r['corpus']['with_text']==len(allowed)==729704
assert r['corpus']['new_documents']+r['corpus']['reused_documents']==len(allowed)
assert r['decision']['doc_len']==1024 and r['total_upper_usd']<=26
write(T/'final_audit.json',dict(queries=1200,top_per_query=1000,exact_partition_merge=True,all_ids_valid=True,
                              all_finite=True,independent_coverage=True,independent_novelty=True,sha=True,budget=True))
print('FINAL_AUDIT_PASS',flush=True)
