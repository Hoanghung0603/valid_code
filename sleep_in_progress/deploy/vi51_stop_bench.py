"""Terminal timeout evidence: STOP after two same bench failures; no retry."""
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parent))
from vi51_local import cli,TEMP,handoff

os.environ['MODAL_PROFILE']=__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>");os.environ['PYTHONIOENCODING']='utf-8'
p=TEMP/'state_<MODAL_PROFILE_A>.json';s=json.loads(p.read_text())
for name in ('bench_a','bench_b'):
    r=s['calls'][name]
    r['status']='failed_terminal_timeout'
    r['terminal_evidence']='Remote subprocess.TimeoutExpired returned by FunctionCall.get; app tasks=0'
    r['function_wall']=json.loads((TEMP/(name+'_wall.json')).read_text())
s['calls']['ping']['mode']='cpu'
s['stop_reason']='Two same bench watchdog timeouts; mandatory STOP, no bulk or respawn'
cli(__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>"),'app','stop','--yes',s['app_id']);s['stopped']=True
s['billing_after']=json.loads(cli(__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>"),'billing','summary','--json'))
s['apps_after']=json.loads(cli(__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>"),'app','list','--json'))
s['gpu_function_seconds']=sum(s['calls'][n]['function_wall']['seconds'] for n in ('bench_a','bench_b'))
p.write_text(json.dumps(s,indent=2),encoding='utf-8');handoff()
print(json.dumps(dict(gpu_function_seconds=s['gpu_function_seconds'],reserved_upper=s['reserved_upper'],billing_after=s['billing_after'],stopped=True)))
