"""Run only the eight approved recovery inputs, at most four H100s."""
from pathlib import Path
import sys
import os
import subprocess
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,forecast,RATE
from vi55_recovery_control import T,MISSING
PY=T.parent/'modalenv/Scripts/python.exe'


def ctl(*args):
    subprocess.run([str(PY),str(ROOT/'deploy/vi55_recovery_control.py'),*args,'--profile',__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>")],check=True,
                   env=dict(os.environ,MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE_D", "<MODAL_PROFILE_D>"),PYTHONIOENCODING='utf-8'))


def run(job,function='work',mode='bulk',parts=(),watchdog=800):
    s=read(T/'recovery_state_<MODAL_PROFILE_D>.json')
    if job not in s['calls']:
        ctl('spawn','--job',job,'--function',function,'--mode',mode,'--watchdog',str(watchdog),'--parts',*parts)
    s=read(T/'recovery_state_<MODAL_PROFILE_D>.json');assert s['calls'][job]['status'] in ('running','done')
    ctl('wait','--job',job)


if __name__=='__main__':
    ping=read(T/'recovery_<MODAL_PROFILE_D>_ping_fixed.json')
    assert set(ping['inputs'])==set(MISSING) and not ping['files']
    run('gate',mode='gate',parts=MISSING[:1],watchdog=100)
    gate=read(T/'recovery_<MODAL_PROFILE_D>_gate.json')
    assert gate['repeat_cosine_min']>=.999 and gate['repeat_cosine_mean']>=.9999
    run('query',mode='query',watchdog=100)
    ctl('download','--remote','/out/query/query_embeddings.npz','--local',str(T/'shared_query_embeddings.npz'))
    prep=read(T/'prepare_report.json')
    pairs=[MISSING[i:i+2] for i in range(0,8,2)]
    throughput=read(T/'<MODAL_PROFILE_C>_bench1024.json')['tokens_per_second']
    plan=[]
    for i,parts in enumerate(pairs):
        tokens=sum(prep['parts'][p]['tokens1024'] for p in parts)
        estimate=forecast(tokens,throughput,jobs=1,overhead=55)
        watchdog=int(estimate['gpu_seconds'])+20
        plan.append(dict(job=f'bulk_recovery{i:02d}',parts=parts,tokens=tokens,watchdog=watchdog,
                         reserve=(watchdog+12)*RATE))
    current=sum(read(p)['reserved'] for p in T.glob('recovery_state_*.json'))
    assert 18.938931953320207+current+sum(p['reserve'] for p in plan)+3*252*RATE+.15<=26
    write(T/'recovery_plan.json',dict(workers=plan,prior_upper=18.938931953320207))
    for worker in plan:
        s=read(T/'recovery_state_<MODAL_PROFILE_D>.json')
        if worker['job'] not in s['calls']:
            ctl('spawn','--job',worker['job'],'--function','work','--mode','bulk','--watchdog',str(worker['watchdog']),
                '--parts',*worker['parts'])
    for worker in plan:ctl('wait','--job',worker['job'])
    run('inventory',function='cpu_ping_inventory')
    inventory=read(T/'recovery_<MODAL_PROFILE_D>_inventory.json')
    assert len(inventory['ids'])==80000 and set(inventory['parts'])==set(MISSING)
    # The loop above may finish after this launcher was started; keep final manifest
    # construction also available as a standalone read-only audit.
    manifest={}
    state=read(T/'recovery_state_<MODAL_PROFILE_D>.json')
    for worker in plan:
        result=state['calls'][worker['job']]['result']
        for part in result['parts']:
            path='/data/out/'+worker['job']+'/'+part['part']+'.emb.npz'
            assert inventory['files'][path]=={k:part['output'][k] for k in ('bytes','sha256')}
            assert part['output']['documents']==10000 and part['output']['shape']==[10000,4096]
            manifest[part['part']]=part
    assert set(manifest)==set(MISSING)
    write(T/'recovery_part_manifest.json',manifest)
    print('RECOVERY_EIGHT_PARTS_DONE',flush=True)
