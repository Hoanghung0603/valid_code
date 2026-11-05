"""Audit completed recovery call metadata against independently read Volume inventory."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write
from vi55_recovery_control import T,MISSING
plan=read(T/'recovery_plan.json')['workers'];inventory=read(T/'recovery_<MODAL_PROFILE_D>_inventory.json')
assert len(inventory['ids'])==80000 and set(inventory['parts'])==set(MISSING)
state=read(T/'recovery_state_<MODAL_PROFILE_D>.json');manifest={};ids=set()
for worker in plan:
    r=state['calls'][worker['job']];assert r['status']=='done'
    for part in r['result']['parts']:
        path='/data/out/'+worker['job']+'/'+part['part']+'.emb.npz'
        audit=part['output']
        assert inventory['files'][path]=={k:audit[k] for k in ('bytes','sha256')}
        assert audit['documents']==10000 and audit['shape']==[10000,4096] and audit['dtype']=='float16'
        assert audit['l2_max_error']<.001 and not ids.intersection(audit['ids'])
        assert part['attempt'] in (1,2)
        ids.update(audit['ids']);manifest[part['part']]=part
assert set(manifest)==set(MISSING) and ids==set(inventory['ids'])
write(T/'recovery_part_manifest.json',manifest)
print('RECOVERY_PART_AUDIT_PASS parts=8 documents=80000',flush=True)
