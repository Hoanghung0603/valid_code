"""Repair log watcher for existing call only: never spawn GPU or write shared state."""
import json
import sys
import threading
import modal
from vi41_local import TEMP,STATE,cmd
from vi41_resume import monitor

phase=sys.argv[1]
r=json.loads(STATE.read_text())['calls'][phase]
event=threading.Event()
p=monitor(r['app_id'],r['call_id'],event,TEMP/(phase+'_watch.log'),r['started'])
try:
    result=modal.FunctionCall.from_id(r['call_id']).get(timeout=r.get('seconds',600)+180)
    assert not event.is_set(),'Repeated init failure; stopped'
    print(json.dumps(result))
finally:
    p.terminate()
