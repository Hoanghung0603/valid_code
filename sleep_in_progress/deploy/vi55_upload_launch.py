"""Upload SHA-audited disjoint inputs and launch each immutable planned worker once."""
import argparse
from pathlib import Path
import subprocess
import sys
import os
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,fingerprint
from vi55_local import cli,TEMP,MODAL
a=argparse.ArgumentParser();a.add_argument('--profile',required=True);args=a.parse_args()
os.environ.update(PYTHONIOENCODING='utf-8',MODAL_PROFILE=args.profile)
plan=read(TEMP/'decision.json');prep=read(TEMP/'prepare_report.json')
statepath=TEMP/f'upload_{args.profile}.json';uploaded=read(statepath) if statepath.exists() else {}
for worker in [w for w in plan['workers'] if w['profile']==args.profile]:
    for name in ['queries.json']+worker['parts']:
        p=TEMP/'modal_input'/name
        if name in prep['parts']:assert fingerprint(p)=={k:prep['parts'][name][k] for k in ('bytes','sha256')}
        digest=fingerprint(p)
        if uploaded.get(name)==digest:continue
        cli(args.profile,'volume','put','--force','r2ai-vi55-data',str(p),'/in/'+name)
        uploaded[name]=digest;write(statepath,uploaded);print('UPLOADED '+name,flush=True)
    state=read(TEMP/f'state_{args.profile}.json')
    if worker['job'] in state['calls']:
        print('EXISTING_DO_NOT_RESPAWN '+worker['job'],flush=True);continue
    subprocess.run([sys.executable,str(ROOT/'deploy/vi55_local.py'),'spawn','--profile',args.profile,'--job',worker['job'],
                    '--function','work','--mode','bulk','--watchdog',str(worker['watchdog']),'--doc-len',str(plan['doc_len']),
                    '--parts',*worker['parts']],check=True,env=dict(os.environ,PYTHONIOENCODING='utf-8',MODAL_PROFILE=args.profile))
print('UPLOAD_LAUNCH_DONE '+args.profile,flush=True)
