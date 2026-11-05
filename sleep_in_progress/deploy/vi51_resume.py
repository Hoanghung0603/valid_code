"""Resume only: poll/collect existing call IDs; build each tier at most once."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi51')
CONTROL=ROOT/'deploy/vi51_sdpa_local.py'


def run(*args):
    return subprocess.run([sys.executable,str(CONTROL),*args],cwd=ROOT,
        env=dict(os.environ,PYTHONIOENCODING='utf-8'),check=True)


def load(p):return json.loads(p.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tier',choices=['a','b'],required=True)
    ap.add_argument('--build',action='store_true');ap.add_argument('--wait',type=int,default=45)
    ap.add_argument('--stop',action='store_true')
    a=ap.parse_args()
    workerlock=TEMP/('resume_'+a.tier+'.lock')
    if workerlock.exists():
        pid=int(workerlock.read_text())
        try:os.kill(pid,0)
        except OSError:pass
        else:raise RuntimeError(f'Existing collector pid={pid}; do not duplicate')
    workerlock.write_text(str(os.getpid()))
    while True:
        records=[]
        for p in TEMP.glob('state_sdpa_*.json'):
            state=load(p)
            records.extend((state['profile'],job,r) for job,r in state['calls'].items() if r.get('tier')==a.tier and r.get('mode')=='bulk')
        assert records,'No existing bulk calls; this command NEVER spawns'
        for profile,job,r in records:
            if r['status']=='failed_terminal':raise RuntimeError(f'Terminal failure {profile} {job}: {r.get("error")}')
            if r['status']=='running':
                try:run('poll','--profile',profile,'--job',job,'--wait',str(a.wait))
                except (subprocess.CalledProcessError,OSError) as exc:print('POLL_ERROR '+repr(exc),flush=True)
            elif r['status']=='done' and not r.get('collected'):
                try:run('collect','--profile',profile,'--job',job)
                except (subprocess.CalledProcessError,OSError) as exc:print('COLLECT_ERROR '+repr(exc),flush=True)
        fresh=[load(p) for p in TEMP.glob('state_sdpa_*.json')]
        calls=[r for s in fresh for r in s['calls'].values() if r.get('tier')==a.tier and r.get('mode')=='bulk']
        if all(r['status']=='done' and r.get('collected') for r in calls):break
        time.sleep(10)
    assert sum(len(r['parts']) for r in calls)==29
    if a.stop:
        for p in TEMP.glob('state_sdpa_*.json'):
            s=load(p)
            if not s.get('stopped') and all(r['status'] in ('done','failed_terminal') for r in s['calls'].values()):
                run('stop','--profile',s['profile'])
    out=TEMP.parent/('vi51'+a.tier)
    if not (out/'prebuild_audit.json').exists():
        subprocess.run(['python',str(ROOT/'deploy/vi51_prebuild.py'),a.tier],cwd=ROOT,
            env=dict(os.environ,PYTHONIOENCODING='utf-8'),check=True)
    if a.build:
        marker=out/'build_intent.json'
        if (out/'build_report.json').exists():print('ALREADY_BUILT',flush=True);return
        if marker.exists():
            print('BUILD_ALREADY_STARTED_NO_REBUILD; inspect separate builder',flush=True)
            return
        marker.write_text(json.dumps(dict(time=time.time(),pid=os.getpid())),encoding='utf-8')
        subprocess.run(['python',str(ROOT/'scripts/vi51.py'),'build','--out',str(out),'--tier',a.tier],cwd=ROOT,
            env=dict(os.environ,PYTHONIOENCODING='utf-8'),check=True)
        subprocess.run(['python',str(ROOT/'scripts/vi51_audit.py'),'--out',str(out)],cwd=ROOT,
            env=dict(os.environ,PYTHONIOENCODING='utf-8'),check=True)
        sys.path.insert(0,str(ROOT/'deploy'))
        from vi51_sdpa_local import handoff
        handoff()
        subprocess.run(['python',str(ROOT/'deploy/vi51_mirror_pack.py'),a.tier],cwd=ROOT,
            env=dict(os.environ,PYTHONIOENCODING='utf-8'),check=True)
        print('TIER_READY '+a.tier,flush=True)
    else:print('COLLECT_AUDIT_READY '+a.tier,flush=True)


if __name__=='__main__':main()
