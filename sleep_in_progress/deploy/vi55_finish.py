"""Merge disjoint whole-corpus searches and produce mandatory phase-1 metrics."""
from pathlib import Path
import sys
import time
import numpy as np
import zstandard as zstd
import json
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,metrics,fingerprint,merge_topk,RATE
T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi55')
profiles=[__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),__import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>")]
parts=[read(T/f'search_{p}/top1000.json.zst') for p in profiles]
counts=[read(T/f'search_{p}/rank_counts.json') for p in profiles]
seen=[set(read(T/f'search_{p}/seen_ids.json')) for p in profiles]
assert not seen[0]&seen[1]
allowed=set(map(int,np.load(T/'all_ids.npy')));assert seen[0]|seen[1]==allowed
doclists=read(T/'vi53_doclists.json');exact={};top={}
for q in parts[0]:
    rows=sorted(parts[0][q]+parts[1][q],key=lambda r:(-r[1],r[0]))[:1000]
    assert len(rows)==1000 and len({r[0] for r in rows})==1000
    assert all(np.isfinite(r[1]) and r[0] in allowed for r in rows)
    top[q]=rows;exact[q]={}
    for d in doclists[q]:
        if d in allowed:exact[q][str(d)]=1+sum(c[q][str(d)] for c in counts)
assert len(top)==1200 and set(map(int,top))==set(range(1,1201))
with zstd.open(T/'q3e8b_full_top1000.json.zst','wt',encoding='utf-8') as f:json.dump(top,f)
write(T/'vi53_exact_ranks.json',exact)
m=metrics(top,read(T/'pool.json'),read(T/'dense_current_top200.json'),doclists,exact)
m['overlap_bge_historical_full']=metrics(top,read(T/'pool.json'),read(T/'dense_top200.json'),doclists,exact)
states=[read(T/f'state_{p}.json') for p in profiles]
qfiles=list((T/'bulk_reports').rglob('query_embeddings.npz'))
assert len(qfiles)==8
qhash=[fingerprint(p) for p in qfiles]
assert all(x==qhash[0] for x in qhash),'Whole-corpus search must share identical query embeddings'
costs={}
for s in states:
    gpu=sum(r.get('result',{}).get('function_seconds',0) for r in s['calls'].values() if r['function'] in ('work','retrieve'))
    before=float(s['baseline']['metered_cost']);after=float(s.get('after',{}).get('metered_cost',before))
    pp=s['profile']; planned=[w for w in read(T/'decision.json')['workers'] if w['profile']==pp]
    prep=read(T/'prepare_report.json')
    reuse=read(T/f'{pp}_reuse.json');reused_vi=len(set(reuse['ids'])&allowed)
    costs[s['profile']]=dict(gpu_seconds=gpu,billing_before=before,billing_after=after,snapshot_delta=max(0,after-before),
                             bulk_parts=sum(len(w['parts']) for w in planned),
                             new_documents=sum(prep['parts'][n]['docs'] for w in planned for n in w['parts']),reused_vi_documents=reused_vi,
                             upper_usd=max(s['upper'],after-before+.05),app_stopped=s.get('stopped',False))
report=dict(phase=1,completed=time.time(),corpus=read(T/'prepare_report.json'),decision=read(T/'decision.json'),
            reuse=read(T/'reuse_verified.json'),bench=read(T/'<MODAL_PROFILE_C>_bench1024.json'),metrics=m,costs=costs,
            total_gpu_seconds=sum(c['gpu_seconds'] for c in costs.values()),total_upper_usd=sum(c['upper_usd'] for c in costs.values()),
            output=fingerprint(T/'q3e8b_full_top1000.json.zst'),rank_convention='competition ranks on L2 renormalized fp16 vectors, rank-only cosine rounded1e-6; output cosine unrounded fp32',
            vi53_non_vi_excluded_from_rank=sorted({d for ds in doclists.values() for d in ds}-allowed),
            dense_comparison_source='vi36/dense_v36_top200.json (incremental global merge); historical diag18/dense_full_top400.json also reported',
            gold_labels_used=False,phase2=False,not_checked=['quality/F2 without labels','billing not settled'])
assert report['total_upper_usd']<=26
assert all(c['app_stopped'] for c in costs.values())
assert read(T/'input_audit.json')['exact_ids']
report['input_audit']=read(T/'input_audit.json')
report['query_audit']=dict(workers=8,sha_equal=True,fingerprint=qhash[0])
write(T/'phase1_report.json',report);print(json.dumps(dict(metrics=m,costs=costs,total_upper_usd=report['total_upper_usd'])),flush=True)
