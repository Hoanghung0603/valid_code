"""Reconcile completed output with saved calls after concurrent collectors finish."""
import json
from vi45_bulk_local import TEMP

KNOWN={
    __import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"):{
        'old000':('fc-01M67SGYWJXD9VW445QAJGAK3P',1350),
        'old001':('fc-01M67SK5QS37C17JNG2NY3F8S0',1350)},
    __import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>"):{
        'new006':('fc-01M67STME1ER1CE88N3R7Z8QMY',2000),
        'new010':('fc-01M67SWXZ0MPGMHKZHDGKN1HHW',2000),
        'new014':('fc-01M67T1XKAAPSCFPMP08YTNEKZ',2200)}}

for profile in (__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"),__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>")):
    sp=TEMP/('bulk_state_'+profile+'.json')
    s=json.loads(sp.read_text())
    for name,(ident,watchdog) in KNOWN[profile].items():
        if name not in s['calls']:
            config=json.loads((TEMP/(name+'.job.json')).read_text())
            s['calls'][name]=dict(call_id=ident,watchdog=watchdog,status='running',
                parts=[p['name'] for p in config['parts']],recovered_from='persisted handoff; never respawned')
    for job in (TEMP/'outputs'/profile).glob('*') if (TEMP/'outputs'/profile).exists() else []:
        config=json.loads((TEMP/(job.name+'.job.json')).read_text())
        expected=[p['name'].replace('.jsonl.zst','.logits.json') for p in config['parts']]
        if not all((job/n).exists() for n in expected) or not (job/'function_wall.json').exists():
            continue
        if job.name not in s['calls']:
            raise RuntimeError('Saved call missing; recover ID from handoff before stop')
        s['calls'][job.name]['status']='done'
        s['calls'][job.name]['output_verified_local']=expected
    reserve=0.
    for name,call in s['calls'].items():
        if name=='ping':
            continue
        wall=TEMP/'outputs'/profile/name/'function_wall.json'
        # Keep launch reservation: model/container startup before function entry
        # can exceed the measured function wall. Do not release budget from it.
        reserve+=(call['watchdog']+25)*4.5232/3600
    s['reserved_upper']=reserve
    sp.write_text(json.dumps(s,indent=2))
    print(profile,{k:v['status'] for k,v in s['calls'].items()})
