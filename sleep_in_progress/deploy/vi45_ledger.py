"""Local conservative function-wall ledger; billing snapshots stay separate."""
import json
from pathlib import Path
from vi45_bulk_local import TEMP

profiles={}
for profile in (__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"),__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>")):
    s=json.loads((TEMP/('bulk_state_'+profile+'.json')).read_text())
    seconds=reserve=0.
    parts=pairs=0
    for name,c in s['calls'].items():
        if name=='ping':
            continue
        wall=TEMP/'outputs'/profile/name/'function_wall.json'
        if wall.exists():
            sec=json.loads(wall.read_text())['seconds']
            seconds+=sec
        reserve+=(c['watchdog']+25)*4.5232/3600
        dest=TEMP/'outputs'/profile/name
        for p in dest.glob('*.logits.json'):
            parts+=1
            pairs+=len(json.loads(p.read_text()))
    delta=None
    if 'billing_after' in s:
        delta=float(s['billing_after']['metered_cost'])-float(s['billing_before']['metered_cost'])
    profiles[profile]=dict(completed_function_seconds=seconds,bulk_reserved_upper=reserve,
        total_upper=reserve+(.319106727 if profile==__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>") else 0)+.15,
        completed_parts=parts,pairs=pairs,billing_delta_snapshot=delta,
        billing_before=s['billing_before'],billing_after=s.get('billing_after'))
total=sum(v['total_upper'] for v in profiles.values())
report=dict(profiles=profiles,total_upper=total,budget=12,within_cap=total<=12,
    note='Original watchdog reserve retained, +25s per call and .15 CPU/network per workspace, plus both bench attempts. Pre-function startup not inferred from function wall; not settled billing.')
(TEMP/'bulk_ledger.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
assert total<=12
