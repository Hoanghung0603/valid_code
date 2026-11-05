"""Snapshot cost attribution conservatively; never treat shared history as vi54."""
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).parent))
from vi54_control import cli,handoff

OUT=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi54')
RATE=4.5232/3600


def main():
    result={};total=0
    for p in sorted(OUT.glob('state_*.json')):
        s=json.loads(p.read_text(encoding='utf-8'))
        snapshot=cli(s['profile'],'billing','summary','--json',parse=True)
        baseline=float(s['billing_before']['metered_cost']);current=float(snapshot['metered_cost'])
        gpu=sum(r.get('result',{}).get('function_seconds',0) for r in s['calls'].values() if r['kind'] in ('bench','bulk'))
        # Include 0.1 CPU per workspace and both +8s wrapper reserve/call and
        # pending watchdog reserve in control state's upper; snapshot may lag.
        upper=max(s['reserved_upper'],max(0,current-baseline)+.1)
        result[s['profile']]=dict(snapshot=snapshot,baseline_metered_usd=baseline,
            snapshot_delta_usd=current-baseline,gpu_function_seconds=gpu,gpu_function_usd=gpu*RATE,
            control_reserved_upper_usd=s['reserved_upper'],upper_usd=upper,settled=False)
        total+=upper;s['latest_billing']=result[s['profile']]
        p.write_text(json.dumps(s,indent=2),encoding='utf-8')
    # Prior vi54 failed image build before20:35 is included conservatively as
    # difference from20:32 snapshot to fresh CPU-work baseline (no GPU then).
    prior=sum(max(0,result[p]['baseline_metered_usd']-v) for p,v in
              [(__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),6.64859249),(__import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>"),13.90588422)] if p in result)
    output=dict(workspaces=result,prior_cpu_snapshot_delta_usd=prior,total_upper_usd=total+prior,
                gpu_function_seconds=sum(v['gpu_function_seconds'] for v in result.values()),budget_usd=25)
    (OUT/'cost_ledger.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    handoff();print(json.dumps(output),flush=True)


if __name__=='__main__':main()
