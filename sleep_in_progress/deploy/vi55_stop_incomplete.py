"""Mandatory stop after the same resume assertion failed in two workspaces."""
import os
from pathlib import Path
import sys
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,RATE
from vi55_local import TEMP,cli,MODAL,handoff
for profile,job in [(__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),'bulk03'),(__import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>"),'bulk07')]:
    os.environ.update(PYTHONIOENCODING='utf-8',MODAL_PROFILE=profile)
    sp=TEMP/f'state_{profile}.json';s=read(sp);r=s['calls'][job]
    assert 'CalledProcessError' in r['error']
    r.update(status='failed_terminal',stop_reason='Preemption auto-restarted invocation; same non-resumable existing-output assertion failed twice; mandatory STOP, no respawn')
    s['stop_reason']=r['stop_reason'];write(sp,s)
    dest=TEMP/'bulk_reports'/profile/job;dest.mkdir(parents=True,exist_ok=True)
    for name in ['partial_report.json','function_wall.json','query_embeddings.npz']:
        cli(profile,'volume','get','--force','r2ai-vi55-data',f'/out/{job}/{name}',str(dest/name))
    subprocess.run([str(TEMP.parent/'modalenv/Scripts/python.exe'),str(ROOT/'deploy/vi55_local.py'),'stop','--profile',profile],check=True,
                   env=dict(os.environ,PYTHONIOENCODING='utf-8',MODAL_PROFILE=profile))
states=[read(TEMP/f'state_{p}.json') for p in (__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),__import__("os").environ.get("MODAL_PROFILE_C", "<MODAL_PROFILE_C>"))]
costs={};complete=0;partial=0;known_gpu=0
for s in states:
    finished=[r for r in s['calls'].values() if r.get('mode')=='bulk' and r['status']=='done']
    complete+=sum(r['result']['documents'] for r in finished)
    failed=[(n,r) for n,r in s['calls'].items() if r['status']=='failed_terminal']
    for n,r in failed:
        partial+=sum(x['documents'] for x in read(TEMP/'bulk_reports'/s['profile']/n/'partial_report.json')['parts'])
    known=sum(r.get('result',{}).get('function_seconds',0) for r in s['calls'].values() if r['function']=='work')
    known_gpu+=known
    delta=float(s['after']['metered_cost'])-float(s['baseline']['metered_cost'])
    costs[s['profile']]=dict(snapshot_before=s['baseline'],snapshot_after=s['after'],snapshot_delta=delta,
                             known_success_gpu_seconds=known,upper_usd=max(s['upper'],delta+.05),app_stopped=s['stopped'],
                             failed_calls=[n for n,r in failed],complete_new_documents=sum(r['result']['documents'] for r in finished))
report=dict(phase=1,status='STOPPED_INCOMPLETE',reason='Same resume assertion twice after Modal preemption; user rule requires STOP. No respawn/recovery launches.',
            corpus=read(TEMP/'prepare_report.json'),reuse=read(TEMP/'reuse_verified.json'),decision=read(TEMP/'decision.json'),
            bench=read(TEMP/'<MODAL_PROFILE_C>_bench1024.json'),new_documents_complete_calls=complete,new_documents_partial_calls=partial,
            new_documents_embedded=complete+partial,new_documents_missing=506707-complete-partial,costs=costs,
            known_success_gpu_seconds=known_gpu,total_upper_usd=sum(v['upper_usd'] for v in costs.values()),
            top1000_written=False,metrics=None,metrics_unavailable_reason='Full corpus embedding incomplete; no partial proxy misreported as whole-corpus retrieval',
            tests=dict(passed=1194,skipped=2,vi55_passed=10),not_checked=['whole-corpus top1000 and all requested metrics','failed preempted exact GPU wall','settled billing','gold quality/F2'],
            phase2=False,submission=False)
assert report['total_upper_usd']<=26
write(TEMP/'phase1_report.json',report);print(report,flush=True)
