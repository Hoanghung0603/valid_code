"""Read-only independent final audit; no inference or artifact rewriting."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,fingerprint,metrics,audit_embedding
T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi55')
report=read(T/'phase1_report.json');assert report['status']=='DONE' and not report['phase2'] and not report['submission']
assert fingerprint(T/'q3e8b_full_top1000.json.zst')=={k:report['output'][k] for k in ('bytes','sha256')}
top=read(T/'q3e8b_full_top1000.json.zst');allowed=set(map(int,np.load(T/'all_ids.npy')))
assert set(map(int,top))==set(range(1,1201))
partitions=[read(T/f'search_{p}/top1000.json.zst') for p in (__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),__import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>"),__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>"))]
boundary=[]
for q,rows in top.items():
    assert len(rows)==1000 and len({d for d,_ in rows})==1000
    assert all(d in allowed and np.isfinite(s) and -1.0001<=s<=1.0001 for d,s in rows)
    assert rows==sorted(rows,key=lambda r:(-r[1],r[0]))
    candidate={d:s for p in partitions for d,s in p[q]}
    assert rows==[list(r) for r in sorted(candidate.items(),key=lambda r:(-r[1],r[0]))[:1000]]
audit=audit_embedding(T/'shared_query_embeddings.npz',[r['id'] for r in read(T/'modal_input/queries.json')],4096,'float32')
assert audit==report['query_audit'] and audit['documents']==1200
doclists=read(T/'vi53_doclists.json');pool=read(T/'pool.json');dense=read(T/'dense_current_top200.json')
ranks=read(T/'vi53_exact_ranks.json')
recomputed=metrics(top,pool,dense,doclists,ranks)
assert all(report['metrics'][k]==v for k,v in recomputed.items())
for q,rows in top.items():
    rounded=np.round(np.asarray([s for _,s in rows])*1e6)/1e6
    ascending=np.sort(rounded)
    for index,(d,_) in enumerate(rows):
        if str(d) in ranks[q]:
            assert 1<=ranks[q][str(d)]<=len(allowed)
            # Rank implementation uses independently computed fp64 thresholds.
            # Only compare targets separated from rounding boundaries by >1e-6.
            threshold=rounded[index]
            minimum=1+len(ascending)-int(np.searchsorted(ascending,threshold+1.00001e-6,side='right'))
            maximum=1+len(ascending)-int(np.searchsorted(ascending,threshold-1.00001e-6,side='right'))
            if threshold-rounded[-1]>2e-6:
                assert minimum<=ranks[q][str(d)]<=maximum
            elif ranks[q][str(d)]>1000:
                # A rounded target threshold can move across the unrounded top1000
                # boundary; unseen tail scores cannot be bounded by top1000 alone.
                boundary.append((q,d,ranks[q][str(d)]))
assert report['total_upper_usd']<=26 and report['top1000_audit']['documents']==729704
print('INDEPENDENT_FINAL_AUDIT_PASS queries=1200 top=1000 corpus=729704 query_full=1200 metrics=exact rounded_tail='+str(boundary),flush=True)
