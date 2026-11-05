"""One event stream, no polling/relaunch; emergency stop only on billing denial."""
import json
import os
import subprocess

from vi45_bulk_local import MODAL,TEMP,cli

profile=__import__("os").environ.get("MODAL_PROFILE_A", "<MODAL_PROFILE_A>")
app='ap-66LiHBiFzKxfVq8LWeDZx0'
call='fc-01M67T1XKAAPSCFPMP08YTNEKZ'
env=dict(os.environ,MODAL_PROFILE=profile,PYTHONIOENCODING='utf-8')
p=subprocess.Popen([MODAL,'app','logs',app,'--function-call',call,'--follow','--timestamps'],
    env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace')
finished=set()
try:
    with (TEMP/'tail_monitor.log').open('a',encoding='utf-8') as log:
        for line in p.stdout:
            log.write(line)
            log.flush()
            lowered=line.lower()
            if any(x in lowered for x in ('billing error','insufficient credits','workspace is disabled','payment required')):
                print('BILLING_STOP '+line,flush=True)
                cli(profile,'app','stop','--yes',app)
                break
            if 'PART_DONE ' in line:
                print(line.rstrip(),flush=True)
                try:
                    finished.add(json.loads(line.split('PART_DONE ',1)[1])['part'])
                except (ValueError,KeyError):
                    pass
                if len(finished)==6:
                    print('TAIL_SIX_COMMITTED',flush=True)
                    break
            if 'Traceback' in line or 'TimeoutExpired' in line:
                print(line.rstrip(),flush=True)
finally:
    p.terminate()
    p.wait(timeout=15)
