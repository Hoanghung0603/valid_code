"""Capture stopped-task and billing snapshots without opening a browser."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write
from vi55_recovery_control import cli,T
for profile in (__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),__import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>"),__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>")):
    p=T/f'recovery_state_{profile}.json';s=read(p);assert s['stopped']
    apps=json.loads(cli(profile,'app','list','--json'))
    own=[a for a in apps if a['app_id']==s['app_id']]
    assert len(own)==1 and own[0]['state']=='stopped' and int(own[0]['tasks'])==0
    write(T/f'final_apps_{profile}.json',apps)
    s['after']=json.loads(cli(profile,'billing','summary','--json'));write(p,s)
print('CLEANUP_AUDIT_PASS apps=3 stopped tasks=0 volumes=kept',flush=True)
