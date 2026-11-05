"""Independent exact ID/length/SHA audit of all prepared new document inputs."""
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,iterz,fingerprint
T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi55');p=read(T/'prepare_report.json')
old=set(map(int,np.load(T/'old_ids.npy')));allids=set(map(int,np.load(T/'all_ids.npy')))
seen=set();tokens=0;tokens512=0
for n,m in p['parts'].items():
    path=T/'modal_input'/n;assert fingerprint(path)=={k:m[k] for k in ('bytes','sha256')}
    rows=list(iterz(path));ids=[r['id'] for r in rows]
    assert len(ids)==len(set(ids))==m['docs'] and not seen.intersection(ids) and not old.intersection(ids)
    assert all(0<len(r['input_ids'])<=1024 and all(isinstance(t,int) and t>=0 for t in r['input_ids']) for r in rows)
    n1024=sum(len(r['input_ids']) for r in rows);n512=sum(min(512,len(r['input_ids'])) for r in rows)
    assert n1024==m['tokens1024'] and n512==m['tokens512']
    seen.update(ids);tokens+=n1024;tokens512+=n512
assert seen|old==allids and tokens==p['tokens1024'] and tokens512==p['tokens512']
reuse=[read(T/f'<MODAL_PROFILE_A>{x}_reuse.json') for x in ('b','c')]
historical=set(reuse[0]['ids'])|set(reuse[1]['ids'])
assert old==historical&allids and len(historical)==223065
qs=read(T/'modal_input/queries.json');assert len(qs)==1200 and {q['id'] for q in qs}==set(range(1,1201))
assert all(0<len(q['input_ids'])<=256 for q in qs)
write(T/'input_audit.json',dict(documents=len(allids),new=len(seen),reused=len(old),parts=len(p['parts']),
                              sha_all=True,exact_ids=True,no_overlap=True,query_count=1200,tokens1024=tokens,tokens512=tokens512))
print('INPUT_AUDIT_PASS '+str(len(allids)),flush=True)
