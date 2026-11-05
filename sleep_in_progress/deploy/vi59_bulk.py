"""One controller, explicit spawn then blocking get; durable state resumes without respawn."""
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write
OUT=ROOT.parent/'r2ai_deliver/cand/vi59d'
PY=sys.executable
os.environ.update(PYTHONIOENCODING='utf-8',MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>"))


def cmd(*args):
    subprocess.run([PY,str(ROOT/'deploy/vi59_control.py'),*args],cwd=ROOT,check=True,env=os.environ)


def state():return read(OUT/'state.json')


def wait(job):
    r=state()['calls'][job]
    if r['status']=='running':cmd('poll','--job',job,'--wait','2450')
    assert state()['calls'][job]['status']=='done',('STOP failed/ambiguous',job,state()['calls'][job])


def main():
    lock=OUT/'bulk_controller.lock'
    with lock.open('x',encoding='utf-8') as f:f.write(str(os.getpid()))
    try:
        jobs=[]
        for kind,n,groups,watchdog in [('qwen',13,4,1300),('bge',15,2,1300)]:
            chunks=[[] for _ in range(groups)]
            for i in range(n):chunks[i%groups].append(f'{kind}.part{i:03d}.jsonl.zst')
            jobs.extend((f'{kind}{i:02d}',kind,p,watchdog) for i,p in enumerate(chunks))
        write(OUT/'bulk_plan.json',[dict(job=j,kind=k,parts=p,watchdog=w) for j,k,p,w in jobs])
        for kind in ('qwen','bge'):
            for j,k,p,w in jobs:
                if k!=kind:continue
                if j not in state()['calls']:cmd('spawn','--job',j,'--function',k,'--watchdog',str(w),'--parts',*p)
            for j,k,p,w in jobs:
                if k==kind:wait(j)
            for j,k,p,w in jobs:
                if k==kind:cmd('collect','--job',j)
        cmd('stop')
        write(OUT/'bulk_done.json',dict(done=True))
    finally:
        lock.unlink()


if __name__=='__main__':main()
