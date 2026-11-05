"""Write local budget/call snapshot. Reports known walls, not invented billing deltas."""
import json
from pathlib import Path
import time

TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi51')


def main():
    workspaces={}
    for p in TEMP.glob('state_sdpa_*.json'):
        s=json.loads(p.read_text());calls=s['calls']
        walls=sum(r.get('result',{}).get('function_seconds',0) for r in calls.values())
        if s['profile']==__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"):walls+=149.131412915
        workspaces[s['profile']]=dict(app_id=s.get('app_id'),reserved_upper=s['reserved_upper'],
            known_gpu_function_seconds=walls,part_done=sum(len(r['parts']) for r in calls.values() if r.get('mode')=='bulk' and r['status']=='done'),
            part_assigned=sum(len(r['parts']) for r in calls.values() if r.get('mode')=='bulk'),
            calls={n:dict(id=r.get('call_id'),status=r['status'],tier=r['tier'],parts=r['parts']) for n,r in calls.items()},
            billing_before=s.get('billing_before'),billing_latest=s.get('billing_latest',s.get('billing_after')))
    upper=.584092+sum(x['reserved_upper'] for x in workspaces.values())
    assert upper<=30
    report=dict(time=time.time(),prior_vllm_gpu_function_seconds=425.060119,
        known_gpu_function_seconds=425.060119+sum(x['known_gpu_function_seconds'] for x in workspaces.values()),
        reserved_upper_usd=upper,workspaces=workspaces)
    (TEMP/'ledger.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='workspaces'}),flush=True)


if __name__=='__main__':main()
