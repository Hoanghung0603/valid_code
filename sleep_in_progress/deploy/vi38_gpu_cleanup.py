"""Read-only final app/container snapshots, keeping Volume untouched."""
import json
import os
from pathlib import Path
import subprocess

base = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38')
modal = __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'
os.environ.update(PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
report = {}
for obj in ('app', 'container'):
    raw = subprocess.check_output([modal, obj, 'list', '--json'], text=True, encoding='utf-8')
    report[obj] = json.loads(raw)
assert all(x['state'] == 'stopped' and str(x['tasks']) == '0' for x in report['app'])
assert not report['container']
(base/'cleanup_modal.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print('ALL APPS STOPPED, 0 TASKS, CONTAINERS EMPTY; Volume kept', flush=True)
