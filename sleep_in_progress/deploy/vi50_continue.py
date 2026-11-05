"""Remaining explicit parts, refresh profile inventory each allocation, bounded to two GPUs.

Run only after the initial two collectors finish. Stops on any ambiguous failure.
"""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

from vi50_local import ROOT,TEMP,cli


def invoke(action,profile,*args):
    command=[sys.executable,str(ROOT/'deploy/vi50_local.py'),action,'--profile',profile,*args]
    p=subprocess.run(command,cwd=ROOT,env=dict(os.environ,MODAL_PROFILE=profile,PYTHONIOENCODING='utf-8'),
        capture_output=True,text=True,encoding='utf-8')
    print(p.stdout,flush=True)
    if p.returncode:
        print(p.stderr,flush=True)
        if any(word in (p.stdout+p.stderr).lower() for word in ('billing','credit limit','insufficient credit')):
            sp=TEMP/f'state_{profile}.json'
            if sp.exists():
                state=json.loads(sp.read_text(encoding='utf-8'))
                if state.get('app_id'):
                    cli(profile,'app','stop','--yes',state['app_id'])
        raise RuntimeError(f'STOP {profile}/{action}; inspect saved call/state, never respawn')


def main():
    initial=json.loads((TEMP/'state_<MODAL_PROFILE_A>.json').read_text(encoding='utf-8'))
    assert all(r['status']=='done' for r in initial['calls'].values())
    pending=[f'pairs50.part{i:03d}.jsonl.zst' for i in range(4,16)]
    launched=[]
    while pending or launched:
        if datetime.datetime.now().time()>=datetime.time(21,30):
            raise RuntimeError('Hard stop 21:30; preserve handoff')
        # Always inspect fresh inventory before allocating the next parts.
        profiles=json.loads(cli(__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>"),'profile','list','--json'))
        allowed=[p['name'] for p in profiles if p['name'] in (__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>"),__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),__import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>"))]
        # Fresh workspace first; do not waste/upload unrelated frozen corpora.
        allowed.sort(key=lambda p:(p==__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>"),p))
        for profile in allowed:
            sp=TEMP/f'state_{profile}.json'
            if not sp.exists():
                invoke('deploy',profile)
                invoke('ping',profile)
                invoke('poll',profile,'--job','ping','--wait','180')
            state=json.loads(sp.read_text(encoding='utf-8'))
            active=sum(r['status']=='running' and n!='ping' for n,r in state['calls'].items())
            while pending and active<2:
                names=pending[:2];pending=pending[2:]
                job=('a' if profile==__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>") else 'b' if profile==__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>") else 'c')+names[0].split('part')[1][:3]
                invoke('launch',profile,'--job',job,'--parts',*names,'--watchdog','1500')
                launched.append((profile,job));active+=1
        if launched:
            profile,job=launched.pop(0)
            invoke('poll',profile,'--job',job,'--wait','1800')
            state=json.loads((TEMP/f'state_{profile}.json').read_text(encoding='utf-8'))
            assert state['calls'][job]['status']=='done', 'Timeout; stop without duplicate launch'
            if not pending and not any(p==profile for p,j in launched):
                invoke('stop',profile)
    for sp in TEMP.glob('state_*.json'):
        state=json.loads(sp.read_text(encoding='utf-8'))
        if 'billing_after' not in state:
            invoke('stop',state['profile'])
    print('ALL_BULK_DONE_APPS_STOPPED',flush=True)


if __name__=='__main__':main()
