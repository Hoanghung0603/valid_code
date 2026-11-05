"""Repair state lost by an old concurrent bench poll; NEVER spawn anything."""
import json
from pathlib import Path
import time

P=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi51')
sp=P/'state_sdpa_<MODAL_PROFILE_C>.json';s=json.loads(sp.read_text())
(P/'state_before_id_repair.json').write_text(json.dumps(s,indent=2))
bench=json.loads((P/'outputs/<MODAL_PROFILE_C>/fp16_bench_a/bench_report.json').read_text())
s['calls']['fp16_bench_a'].update(status='done',result=bench,collected=True,finish=time.time(),reserve=(bench['function_seconds']+8)*4.5232/3600)
ids=['fc-01M68PPTBDMZA24A013QEK4K47','fc-01M68PQ0AEAWHCK579GFEDHMD7','fc-01M68PQ69JA5VS6HPQ7QJ2DDAC','fc-01M68PQCJ9AB7GFF14N5R7KKW4']
plan=json.loads((P/'fp16_plan_a.json').read_text())
for j,cid in zip(plan['jobs'],ids):
    if j['job'] not in s['calls']:
        s['calls'][j['job']]=dict(j,status='running',mode='bulk',call_id=cid,start=1793441096.4)
if 'fp16_bulk_b_00' in s['calls']:
    s['calls']['fp16_bulk_b_00'].update(status='running',call_id='fc-01M68PT5YFP2TMAASX5JJH13P0')
s['reserved_upper']=.02+sum(r.get('reserve',0) for r in s['calls'].values())
sp.write_text(json.dumps(s,indent=2),encoding='utf-8')
print('RESTORED_NO_RESPAWN '+str(s['reserved_upper']))
