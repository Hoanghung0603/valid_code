"""Resume known calls only, verify artifacts, launch one partition search, stop own app."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,fingerprint
from vi55_local import cli,TEMP
CONTROL_PYTHON=TEMP.parent/'modalenv/Scripts/python.exe'
a=argparse.ArgumentParser();a.add_argument('--profile',required=True);args=a.parse_args()
os.environ.update(MODAL_PROFILE=args.profile,PYTHONIOENCODING='utf-8')
def control(*cmd):
    subprocess.run([str(CONTROL_PYTHON),str(ROOT/'deploy/vi55_local.py'),*cmd,'--profile',args.profile],check=True,
                   env=dict(os.environ,MODAL_PROFILE=args.profile,PYTHONIOENCODING='utf-8'))
plan=read(TEMP/'decision.json');workers=[w for w in plan['workers'] if w['profile']==args.profile]
state=read(TEMP/f'state_{args.profile}.json')
assert all(w['job'] in state['calls'] for w in workers),'Wait upload-launch completion; do not guess missing calls'
for w in workers:
    state=read(TEMP/f'state_{args.profile}.json');r=state['calls'][w['job']]
    if r['status']=='running':control('poll','--job',w['job'],'--wait','3600')
    state=read(TEMP/f'state_{args.profile}.json');r=state['calls'][w['job']];assert r['status']=='done'
    assert set(x['part'] for x in r['result']['parts'])==set(w['parts'])
    # Verify persisted output locally only after terminal success; keep all embeddings on Volume.
    out=TEMP/'bulk_reports'/args.profile/w['job'];out.mkdir(parents=True,exist_ok=True)
    for name in ['bulk_report.json','function_wall.json','query_embeddings.npz']:
        target=out/name
        if not target.exists():cli(args.profile,'volume','get','--force','r2ai-vi55-data',f"/out/{w['job']}/{name}",str(target))
    assert all(np.isfinite(np.load(out/'query_embeddings.npz')['emb']).ravel())
query_fingerprints=[fingerprint(TEMP/'bulk_reports'/args.profile/w['job']/'query_embeddings.npz') for w in workers]
assert all(m==query_fingerprints[0] for m in query_fingerprints),'Query embeddings differ across workers'
write(TEMP/f'query_audit_{args.profile}.json',dict(workers=len(workers),sha_equal=True,fingerprint=query_fingerprints[0]))
os.environ['MODAL_IMAGE_BUILDER_VERSION']='2025.06'
cli(args.profile,'deploy','deploy/modal_vi55.py')
allowed=list(map(int,np.load(TEMP/'all_ids.npy')));doclists=read(TEMP/'vi53_doclists.json')
spec=TEMP/f'retrieval_spec_{args.profile}.json';write(spec,dict(allowed_ids=allowed,doclists=doclists))
qpath=TEMP/'bulk_reports'/args.profile/workers[0]['job']/'query_embeddings.npz'
qz=np.load(qpath);target=np.load(TEMP/'vi53_target_embeddings.npz');target_ids=list(map(int,target['ids']));emb=target['emb']
lookup={d:i for i,d in enumerate(target_ids)};thresholds={};allowed_set=set(allowed)
for qi,q in enumerate(qz['ids']):
    ids=[d for d in doclists[str(q)] if d in allowed_set]
    v=emb[[lookup[d] for d in ids]].astype(np.float64);v/=np.linalg.norm(v,axis=1)[:,None]
    query=qz['emb'][qi].astype(np.float64);query/=np.linalg.norm(query)
    scores=np.round((v@query)*1e6)/1e6
    thresholds[str(q)]={str(d):float(s) for d,s in zip(ids,scores)}
tp=TEMP/f'vi53_thresholds_{args.profile}.json';write(tp,thresholds)
for source,dest in [(spec,'/retrieval_spec.json'),(qpath,'/query_embeddings.npz'),(tp,'/vi53_thresholds.json')]:
    assert source.exists();cli(args.profile,'volume','put','--force','r2ai-vi55-data',str(source),dest)
state=read(TEMP/f'state_{args.profile}.json')
if 'search' not in state['calls']:control('spawn','--job','search','--function','retrieve','--mode','search','--watchdog','240')
state=read(TEMP/f'state_{args.profile}.json')
if state['calls']['search']['status']=='running':control('poll','--job','search','--wait','3600')
state=read(TEMP/f'state_{args.profile}.json');assert state['calls']['search']['status']=='done'
dest=TEMP/f'search_{args.profile}';dest.mkdir(exist_ok=True)
for name,meta in state['calls']['search']['result']['outputs'].items():
    path=dest/name
    if not path.exists() or fingerprint(path)!=meta:
        cli(args.profile,'volume','get','--force','r2ai-vi55-data','/search/search/'+name,str(path))
    assert fingerprint(path)==meta
if not state.get('stopped'):control('stop')
print('COLLECT_SEARCH_STOP_DONE '+args.profile,flush=True)
