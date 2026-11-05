"""Local one-shot finalization, only after durable bulk/collect/stop completion."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi56 import TEMP, read, write
from vi56_control import handoff


def main():
    out=TEMP/'vi56';s=read(out/'state.json')
    assert s['stopped']
    plan=read(out/'bulk_plan.json')
    assert all(s['calls'][j[0]]['status']=='done' for j in plan['jobs'])
    assert all((out/'outputs'/j[0]/'report.json').exists() for j in plan['jobs'])
    assert not (out/'build_intent.json').exists(), 'ONE build only'
    subprocess.run([sys.executable,str(ROOT/'scripts/vi56.py'),'build'],check=True)
    subprocess.run([sys.executable,str(ROOT/'deploy/vi56_final_audit.py')],check=True)
    s=read(out/'state.json')
    known=sum(r.get('result',{}).get('function_seconds',0) for r in s['calls'].values() if r['function'] in ('qwen','bge'))
    known+=sum(read(out/n)['seconds'] for n in ('original_qwen_wall.json','original_bge_wall.json'))
    apps=read(out/'apps_final.json')
    assert any(a['app_id']==s['app_id'] and a['state']=='stopped' and int(a['tasks'])==0 for a in apps)
    metered=float(s['after']['metered_cost']); baseline=float(s['baseline']['metered_cost'])
    costs=dict(known_gpu_function_seconds=known,usd_upper=s['upper'],metered_snapshot=metered,
               baseline_snapshot=baseline,snapshot_delta=metered-baseline,billing_settled=False,
               all_calls_terminal=True,app_stopped=True,tasks=0,volumes_retained=True)
    costs['usd_upper']=max(costs['usd_upper'],costs['snapshot_delta']+.1)
    assert costs['usd_upper']<=10
    write(out/'cost_report.json',costs)
    s['upper']=costs['usd_upper'];s['cost_report']=costs
    s['completed_local_at']=time.time()
    write(out/'state.json',s);handoff(s)
    report=dict(status='LOCAL_COMPLETE_MIRROR_PENDING',build=read(out/'build_report.json'),
                audit=read(out/'independent_audit.json'),offline=read(out/'offline.json'),cost=costs)
    write(out/'final_report.json',report)
    print('VI56_LOCAL_COMPLETE '+json.dumps(report,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
