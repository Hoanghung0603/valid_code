"""Forecast padded SDPA runtime conservatively, balance 29 parts by token count."""
import json
import math
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vi48 import read,write

OUT=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi54')
RATE=4.5232/3600


def main():
    states={p.stem.removeprefix('state_'):read(p) for p in OUT.glob('state_*.json')}
    bench=[r['result'] for s in states.values() for r in s['calls'].values() if r['kind']=='bench' and r['status']=='done']
    assert len(bench)==1
    b=bench[0];assert b['repeat_cosine_min']>=.999 and b['repeat_cosine_mean']>=.9999 and b['sanity_pass']
    pings=[s['calls']['ping']['result'] for s in states.values() if s['calls'].get('ping',{}).get('status')=='done']
    assert pings and all(p['parts']==pings[0]['parts'] for p in pings)
    parts=pings[0]['parts'];assert len(parts)==29 and pings[0]['documents']==223065
    # 30c has ~14USD historical use; 30b ~6.7USD. Allocate 4:3 to keep both
    # safely under their 30USD credit balances, while <=4 H100/workspace.
    bins=[dict(profile=p,job=f'bulk_{i:02d}',parts=[],tokens=0) for p in sorted(states)
          for i in range(4 if p==__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>") else 3)]
    for name,meta in sorted(parts.items(),key=lambda kv:-kv[1]['tokens']):
        target=min(bins,key=lambda b:b['tokens']);target['parts'].append(name);target['tokens']+=meta['tokens']
    overhead=max(45,b['function_seconds']-b['seconds'])
    for job in bins:
        job['forecast_seconds']=1.3*(job['tokens']/b['tokens_per_second']+overhead+30)
        job['watchdog']=math.ceil(job['forecast_seconds'])
        assert job['watchdog']<=3600
        job['reserved_usd']=(job['watchdog']+8)*RATE
    ledger=read(OUT/'cost_ledger.json') if (OUT/'cost_ledger.json').exists() else {}
    prior=max(sum(s['reserved_upper'] for s in states.values()),ledger.get('total_upper_usd',0))
    result=dict(tokens=pings[0]['tokens'],tokens_per_second=b['tokens_per_second'],overhead=overhead,
                margin=1.3,prior_upper_usd=prior,bulk_upper_usd=sum(j['reserved_usd'] for j in bins),jobs=bins)
    result['upper_usd']=prior+result['bulk_upper_usd']
    result['parallel_minutes']=max(j['forecast_seconds'] for j in bins)/60
    result['bulk_allowed']=result['upper_usd']<=25
    # Reserve snapshot-vs-ledger excess too, not merely display it in forecast.
    snapshot_extra=max(0,prior-sum(s['reserved_upper'] for s in states.values()))
    if snapshot_extra:
        states[__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>")]['reserved_upper']+=snapshot_extra
    write(OUT/'forecast.json',result)
    for profile,s in states.items():
        s['forecast']=dict(total_upper_usd=result['upper_usd'],bulk_allowed=result['bulk_allowed'],
                           workspace_upper_usd=s['reserved_upper']+sum(j['reserved_usd'] for j in bins if j['profile']==profile))
        write(OUT/f'state_{profile}.json',s)
    sys.path.insert(0,str(Path(__file__).parent));from vi54_control import handoff
    handoff();print(json.dumps(result),flush=True)
    if not result['bulk_allowed']:raise SystemExit('STOP: forecast >25USD; no bulk')


if __name__=='__main__':main()
