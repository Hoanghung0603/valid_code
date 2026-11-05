"""Forecast and persist a balanced 29-part launch plan, no remote actions."""
import json
import math
import os
from pathlib import Path
import subprocess
import sys

TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi51')
RATE=4.5232/3600


def plan(prep,bench,tier,workers):
    parts=prep['parts'];slots=[[] for _ in range(workers)];tokens=[0]*workers
    for name,r in sorted(parts.items(),key=lambda item:-item[1]['tokens']):
        i=min(range(workers),key=lambda j:tokens[j]);slots[i].append(name);tokens[i]+=r['tokens']
    rate=bench['tokens_per_second']
    jobs=[]
    for i,(names,n) in enumerate(zip(slots,tokens)):
        seconds=math.ceil(1.3*(n+prep['query_tokens'])/rate+90)
        jobs.append(dict(job=f'fp16_bulk_{tier}_{i:02}',tier=tier,parts=sorted(names),tokens=n,watchdog=seconds,reserve=(seconds+8)*RATE))
    return dict(tier=tier,tokens_per_second=rate,jobs=jobs,reserve=sum(j['reserve'] for j in jobs),max_watchdog=max(j['watchdog'] for j in jobs))


if __name__=='__main__':
    if len(sys.argv)>1:
        tier=sys.argv[1];profile=sys.argv[2];indices=list(map(int,sys.argv[3:]))
        p=TEMP/f'fp16_plan_{tier}.json';data=json.loads(p.read_text())
        statepath=TEMP/f'state_sdpa_{profile}.json'
        for i in indices:
            j=data['jobs'][i];state=json.loads(statepath.read_text())
            if j['job'] in state['calls']:
                print('EXISTING_NO_RESPAWN '+j['job'],flush=True);continue
            assert j['watchdog']<=3600
            subprocess.run([sys.executable,str(Path(__file__).with_name('vi51_sdpa_local.py')),'launch',
                '--profile',profile,'--tier',tier,'--mode','bulk','--job',j['job'],'--watchdog',str(j['watchdog']),
                '--parts',*j['parts']],check=True,env=dict(os.environ,MODAL_PROFILE=profile,PYTHONIOENCODING='utf-8'))
        raise SystemExit(0)
    prep=json.loads((TEMP/'prepare_report.json').read_text())
    state=json.loads((TEMP/'state_sdpa_<MODAL_PROFILE_C>.json').read_text())
    reports={t:state['calls']['fp16_bench_'+t]['result'] for t in 'ab'}
    plans={t:plan(prep,reports[t],t,4 if t=='a' else 6) for t in 'ab'}
    path=TEMP/'fp16_plan.json';path.write_text(json.dumps(plans,indent=2),encoding='utf-8')
    print(json.dumps(plans,indent=2),flush=True)
