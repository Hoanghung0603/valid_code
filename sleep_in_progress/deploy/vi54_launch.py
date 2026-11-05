"""Launch exactly forecasted parts once; serial state writes across profiles."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
OUT=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi54')


def main():
    plan=json.loads((OUT/'forecast.json').read_text());assert plan['bulk_allowed'] and plan['upper_usd']<=25
    assigned=[p for j in plan['jobs'] for p in j['parts']]
    assert len(assigned)==len(set(assigned))==29
    for job in plan['jobs']:
        p=OUT/f"state_{job['profile']}.json";state=json.loads(p.read_text())
        if job['job'] in state['calls']:
            r=state['calls'][job['job']]
            assert r['parts']==job['parts'] and r['status'] in ('running','done') and r.get('call_id')
            print('EXISTING_CALL '+r['call_id'],flush=True);continue
        subprocess.run([sys.executable,'deploy/vi54_control.py','spawn','--profile',job['profile'],
            '--job',job['job'],'--kind','bulk','--watchdog',str(job['watchdog']),'--parts',*job['parts']],
            cwd=ROOT,env=dict(os.environ,MODAL_PROFILE=job['profile'],PYTHONIOENCODING='utf-8'),check=True)
    print('ALL_29_PARTS_SPAWNED_ONCE',flush=True)


if __name__=='__main__':main()
