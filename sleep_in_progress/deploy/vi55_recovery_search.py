"""Shared full-query artifact and explicitly audited disjoint retrieval partitions."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,fingerprint,audit_embedding
from vi55_recovery_control import cli,T
PY=T.parent/'modalenv/Scripts/python.exe'


def control(profile,*args):
    subprocess.run([str(PY),str(ROOT/'deploy/vi55_recovery_control.py'),*args,'--profile',profile],check=True,
                   env=dict(os.environ,MODAL_PROFILE=profile,PYTHONIOENCODING='utf-8'))


def prepare_shared():
    p=T/'shared_query_embeddings.npz'
    expected=[r['id'] for r in read(T/'modal_input/queries.json')]
    audit=audit_embedding(p,expected,4096,'float32')
    assert len(expected)==1200 and set(expected)==set(range(1,1201))
    write(T/'shared_query_audit.json',audit)
    qz=np.load(p);target=np.load(T/'vi53_target_embeddings.npz')
    lookup={int(d):i for i,d in enumerate(target['ids'])};emb=target['emb']
    allowed=set(map(int,np.load(T/'all_ids.npy')));doclists=read(T/'vi53_doclists.json');thresholds={}
    for qi,q in enumerate(qz['ids']):
        ids=[d for d in doclists[str(q)] if d in allowed]
        v=emb[[lookup[d] for d in ids]].astype(np.float64);v/=np.linalg.norm(v,axis=1)[:,None]
        query=qz['emb'][qi].astype(np.float64);query/=np.linalg.norm(query)
        thresholds[str(q)]={str(d):float(s) for d,s in zip(ids,np.round((v@query)*1e6)/1e6)}
    write(T/'shared_vi53_thresholds.json',thresholds)


def search(profile):
    inv=read(T/f'recovery_{profile}_inventory.json')
    files=inv['files'];allowed=list(map(int,np.load(T/'all_ids.npy')))
    spec=T/f'recovery_spec_{profile}.json'
    write(spec,dict(allowed_ids=allowed,doclists=read(T/'vi53_doclists.json'),files=files,
                    query=fingerprint(T/'shared_query_embeddings.npz')))
    for source,dest in [(spec,'/retrieval_spec.json'),(T/'shared_query_embeddings.npz','/query_embeddings.npz'),
                        (T/'shared_vi53_thresholds.json','/vi53_thresholds.json')]:
        cli(profile,'volume','put','--force','r2ai-vi55-data',str(source),dest)
    s=read(T/f'recovery_state_{profile}.json')
    if 'search' not in s['calls']:control(profile,'spawn','--job','search','--function','retrieve','--watchdog','240')
    control(profile,'wait','--job','search')
    s=read(T/f'recovery_state_{profile}.json');result=s['calls']['search']['result']
    dest=T/f'search_{profile}';dest.mkdir(exist_ok=True)
    for name,meta in result['outputs'].items():
        path=dest/name
        if not path.exists() or fingerprint(path)!=meta:
            cli(profile,'volume','get','--force','r2ai-vi55-data','/search/search/'+name,str(path))
        assert fingerprint(path)==meta
    control(profile,'stop')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['shared','search']);ap.add_argument('--profile')
    a=ap.parse_args()
    if a.action=='shared':prepare_shared()
    else:search(a.profile)
