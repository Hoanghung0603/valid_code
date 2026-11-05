"""Get completed stage artifacts only; never launches or retries a GPU call."""
import hashlib
import json
import sys
from vi41_local import TEMP,STATE,cmd

phase=sys.argv[1]
state=json.loads(STATE.read_text())
assert state['calls'][phase+'_v2']['status']=='done'
names={'rr':['rr_new.json','rr_function_report.json'],
       'docwin':['docwin_new.json.zst','docwin_function_report.json'],
       'q8b':['q8b_new.json','q8b_function_report.json']}[phase]
for name in names:
    cmd('volume','get','--force','r2ai-runner','/out/vi41/'+name,str(TEMP/name))
print(json.dumps([{'name':n,'bytes':(TEMP/n).stat().st_size,
                  'sha256':hashlib.sha256((TEMP/n).read_bytes()).hexdigest()} for n in names],indent=2))
