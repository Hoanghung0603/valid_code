"""Transfer explicitly named own vi41 export files, verify size/SHA before Volume put."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from vi41_local import TEMP,cmd


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['manifest','put','budget'])
    ap.add_argument('--name',default='')
    a=ap.parse_args()
    if a.stage=='manifest':
        # Run this command on b660 through its normal isolated CPU task, not here.
        raise SystemExit('Use scripts/vi41.py manifest on b660')
    if a.stage=='put':
        report=json.loads((TEMP/'transfer_manifest.json').read_text())
        for row in report['files']:
            p=TEMP/'modal_input'/row['name']
            assert p.stat().st_size==row['bytes'],p
            assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
        for row in report['files']:
            if a.name and row['name']!=a.name:
                continue
            p=TEMP/'modal_input'/row['name']
            cmd('volume','put','--force','r2ai-runner',str(p),'/in/vi41/'+p.name)
        selected=[r for r in report['files'] if not a.name or r['name']==a.name]
        assert selected,'Requested transfer file not in manifest'
        remote=json.loads(cmd('volume','ls','r2ai-runner','/in/vi41','--json'))
        sizes={Path(r['filename']).name:r['size'] for r in remote}
        assert all(sizes.get(r['name'])==r['bytes'] for r in selected)
        (TEMP/'volume_put_report.json').write_text(json.dumps(dict(files=selected,volume_sizes_ok=True,
            source_sha_ok=True,remote_content_sha='not independently downloaded'),indent=2))
        print(json.dumps({'files':len(selected),'bytes':sum(r['bytes'] for r in selected),'sha_ok':True,'only':a.name}))
        return
    exp=json.loads((TEMP/'export_report.json').read_text())
    # Keep upper + every remaining stage forecast including 30% plus cold start.
    n,w=exp['pairs'],exp['windows']
    seconds={'rr':max(180,int(n/200*1.3)+90),
             'docwin':max(180,int(w*2377/1116226*1.3)+90)}
    # Preserve original planned budgets for already-launched stages; audit trail lives in state.
    statepath=TEMP/'state.json'
    calls=json.loads(statepath.read_text()).get('calls',{})
    for stage in tuple(seconds):
        if stage+'_v2' in calls:
            seconds[stage]=calls[stage+'_v2']['seconds']
    if (TEMP/'qselect_report.json').exists():
        q=json.loads((TEMP/'qselect_report.json').read_text())['pairs']
    else:
        # Stage-gated forecast, NOT candidate truncation: exact pre-8B selection
        # must fit this future budget or execution stops before 8B. No subset is scored.
        q=min(n,12000)
    seconds['q8b']=max(180,int(q/9*1.3)+90)
    if 'q8b_v2' in calls:
        seconds['q8b']=calls['q8b_v2']['seconds']
    plan={k:{'seconds':v,'reserve':(v+120)*2.3952/3600+.03} for k,v in seconds.items()}
    (TEMP/'score_budget.json').write_text(json.dumps(plan,indent=2))
    (TEMP/'score_budget_basis.json').write_text(json.dumps({'new_pairs':n,'windows':w,
        'q8b_planning_cap':q,'exact_selection_required_before_q8b':True},indent=2))
    print(json.dumps({'plan':plan,'pairs':n,'windows':w,'q8b_pairs_upper':q,
                      'remaining_reserve':sum(v['reserve'] for v in plan.values())},indent=2))


if __name__=='__main__':
    main()
