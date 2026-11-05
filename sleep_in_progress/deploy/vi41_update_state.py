"""Persist current measured billing and reserve without losing earlier failed-call costs."""
import json
from vi41_local import TEMP,STATE,cmd,save

state=json.loads(STATE.read_text())
# Never discount the historical rewrite/import-restart upper (2.988641667 USD).
# Successful new calls have durable client spawn-intent..result bounds encompassing
# queue, cold-start, whole function, transport; add 60s for scaledown/clock uncertainty.
for key,hourly in [('search_v3',4.7626),('bench_v2',2.3952),('rr_v2',2.3952),
                   ('docwin_v2',2.3952),('q8b_v2',2.3952)]:
    r=state['calls'].get(key)
    if r and r['status']=='done':
        r.setdefault('original_reserve',r['reserve'])
        wall=r['finished']-r['started']
        r['reserve']=min(r['original_reserve'],(wall+60)*hourly/3600+.03)
        r['upper_basis']='entire client intent-to-result wall +60s startup/scaledown uncertainty +.03 USD; prior failed costs unchanged'
        r['client_wall_seconds']=wall
snapshot=json.loads(cmd('billing','summary','--json'))
state.update(billing_after=snapshot,
    snapshot_delta=float(snapshot['metered_cost'])-float(state['billing_before']['metered_cost']),
    total_upper_reserved_usd=sum(r['reserve'] for r in state['calls'].values()))
state['known_modal_gpu_function_seconds']=428.387142+sum(r['result'].get('seconds',0)
    for k,r in state['calls'].items() if k in ('search_v3','bench_v2','rr_v2','docwin_v2','q8b_v2') and r['status']=='done')
state['final_billing_metered_snapshot']=snapshot['metered_cost']
save(state)
print(json.dumps({k:state[k] for k in ('snapshot_delta','total_upper_reserved_usd','known_modal_gpu_function_seconds')}))
