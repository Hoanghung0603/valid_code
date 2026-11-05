"""Resume only persisted jobs, collect once, audit, build once; never launches GPU."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi54')
sys.path.insert(0,str(ROOT/'scripts'))
from vi48 import read,write
from dwmax_diag import owned_lock


def run(*args):
    return subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,
        env=dict(os.environ,PYTHONIOENCODING='utf-8'),check=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--build',action='store_true');ap.add_argument('--stop',action='store_true');a=ap.parse_args()
    with owned_lock(OUT/'resume.lock',f'vi54 resume pid={os.getpid()}'):
        while True:
            running=False
            for p in sorted(OUT.glob('state_*.json')):
                s=read(p)
                for name,r in s['calls'].items():
                    if r['kind']!='bulk':continue
                    if r['status']=='running':
                        try:run('deploy/vi54_control.py','poll','--profile',s['profile'],'--job',name,'--wait','0')
                        except subprocess.CalledProcessError:
                            fresh=read(p)['calls'][name]
                            if fresh['status']=='failed_terminal':raise
                            # Transport ambiguity: retain call id and retry polling only.
                        running=read(p)['calls'][name]['status']=='running' or running
                    fresh=read(p)['calls'][name]
                    assert fresh['status'] in ('running','done'),fresh
                    if fresh['status']=='done' and not fresh.get('collected'):
                        run('deploy/vi54_control.py','collect','--profile',s['profile'],'--job',name)
            if not running:break
            time.sleep(30)
        subprocess.run(['python','scripts/vi54_build.py','collect'],cwd=ROOT,check=True)
        if a.stop:
            for p in sorted(OUT.glob('state_*.json')):
                s=read(p)
                if not s.get('stopped'):run('deploy/vi54_control.py','stop','--profile',s['profile'])
        if a.build:
            if not (OUT/'build_report.json').exists():
                assert not (OUT/'build_intent.json').exists(),'No second build after intent'
                # Modal venv has no scipy/pyarrow; system Python is the CPU pipeline env.
                subprocess.run(['python','scripts/vi54_build.py','build'],cwd=ROOT,check=True)
            subprocess.run(['python','scripts/vi54_audit.py'],cwd=ROOT,check=True)
        write(OUT/'resume_report.json',dict(done=True,build=a.build,stop=a.stop))
        from vi54_control import handoff
        handoff();print('VI54_RESUME_DONE',flush=True)


if __name__=='__main__':main()
