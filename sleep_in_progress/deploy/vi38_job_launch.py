"""One vi38b job, startup-inclusive conservative USD watchdog; never retries."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
BASE = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38')
MODAL = __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'
RATE = (3.95+4*.0473+48*.008)/3600

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['bulk', 'confirm'])
    ap.add_argument('--spent', type=float, default=0.)
    a = ap.parse_args()
    os.environ.update(PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
    dest = BASE/(a.mode+'_out')
    dest.mkdir(exist_ok=True)
    audit = dest/'launch_audit.json'
    assert not audit.exists(), 'STOP launch audit already exists; no automatic relaunch'
    listed = subprocess.check_output([MODAL, 'app', 'list', '--json'], text=True, encoding='utf-8')
    (dest/'apps_before.json').write_text(listed, encoding='utf-8')
    assert all(x.get('State', x.get('state', '')).lower() == 'stopped' for x in json.loads(listed)), listed
    # Leave $2.1 for full-text confirmation and cleanup; include all client cold-start.
    remaining = min(5.9, 8-a.spent) if a.mode == 'bulk' else 8-a.spent
    assert remaining > .3
    seconds = int((remaining-.12)/RATE)-60
    audit.write_text(json.dumps(dict(state='launching', spent=a.spent, ceiling=8)), encoding='utf-8')
    start = time.monotonic()
    timed_out = False
    with (dest/'modal_job.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([MODAL, 'run', 'deploy/modal_llmj.py', '--mode', a.mode,
                                 '--budget', str(remaining-.12), '--spent', '0'],
                                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            code = proc.wait(timeout=20)
    text = (dest/'modal_job.log').read_text(encoding='utf-8')
    appids = list(dict.fromkeys(re.findall(r'\bap-[A-Za-z0-9]+', text)))
    stops = []
    for appid in appids:
        ob = subprocess.run([MODAL, 'app', 'stop', appid, '--yes'], capture_output=True,
                            text=True, encoding='utf-8', timeout=40)
        stops.append(dict(app_id=appid, exit=ob.returncode, stdout=ob.stdout, stderr=ob.stderr))
    wall = time.monotonic()-start
    report = dict(exit=code, timed_out=timed_out, client_wall_s=wall,
                  all_wall_h100_upper_usd=wall*RATE, reserve_egress_usd=.12,
                  cumulative_upper_usd=a.spent+wall*RATE+.12, app_ids=appids, stops=stops,
                  note='Not settled billing; ALL client wall charged at H100+CPU+RAM incl cold-start')
    audit.write_text(json.dumps(report, indent=2), encoding='utf-8')
    assert a.spent+wall*RATE+.12 < 8, 'STOP hard turn budget accounting reached 8USD'
    print(json.dumps(report), flush=True)
    sys.exit(code if code else int(timed_out))

if __name__ == '__main__':
    main()
