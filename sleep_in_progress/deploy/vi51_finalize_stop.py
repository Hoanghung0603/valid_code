"""Record no-GPU workspace cancellation and final bounded ledger."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from vi51_local import cli,TEMP,handoff

p=TEMP/'state_<MODAL_PROFILE_B>.json';s=json.loads(p.read_text())
s.update(app_id='ap-FFr7Wjn6opYf2eY4aHfXMy',stopped=True,
    stop_reason='Deploy initialization stopped after mandatory two bench failures on <MODAL_PROFILE_A>; no calls/GPU')
s['apps_after']=json.loads(cli(__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),'app','list','--json'))
assert any(r['app_id']==s['app_id'] and r['state']=='stopped' and r['tasks']=='0' for r in s['apps_after'])
s['billing_after']=json.loads(cli(__import__("os").environ.get("MODAL_PROFILE_B", "<MODAL_PROFILE_B>"),'billing','summary','--json'))
p.write_text(json.dumps(s,indent=2),encoding='utf-8')
handoff()
report=dict(status='STOP_TWO_BENCH_TIMEOUTS',gpu_function_seconds=425.060119085,
    r2ai_30_upper=.569092,r2ai_30b_upper=.015,total_upper=.584092,
    billed_cost_attributable_snapshot='not separable from concurrent vi50; workspace snapshots in state JSON',
    bench_docs=2000,bench_tokens=2337637,throughput=None,repeat_cosine=None,bulk_forecast=None,
    bulk_docs=0,bulk_pairs=0,zip_vi51a=None,zip_vi51b=None,tests_selected=48,
    input_audit=json.loads((TEMP/'input_audit.json').read_text()),
    cleanup='both own apps stopped/tasks0; Volumes retained; b660 prepare task deleted, lock absent, own PID absent')
(TEMP/'stop_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
