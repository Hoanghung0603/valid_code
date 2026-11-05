"""Single approved bench; client watchdog includes image build and cold start.

No automatic retries. Upper bound charges H100+4CPU+48GiB for ALL client wall,
including CPU-only build/scheduling, with extra reserve for cleanup/network.
"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEST = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38/bench_out2')
MODAL = __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'
RATE = (3.95 + 4*.0473 + 48*.008)/3600


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    os.environ.update(PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
    audit = DEST/'launch_audit.json'
    if audit.exists():
        raise RuntimeError('STOP launch audit already exists; no automatic relaunch')
    listed = subprocess.check_output([MODAL, 'app', 'list', '--json'], text=True, encoding='utf-8')
    (DEST/'apps_before.json').write_text(listed, encoding='utf-8')
    assert not any(a.get('state', a.get('State', '')).lower() not in ('stopped',) for a in json.loads(listed)), listed
    start = time.monotonic()
    timed_out = False
    with (DEST/'modal_bench.log').open('w', encoding='utf-8') as log:
        proc = subprocess.Popen([MODAL, 'run', 'deploy/modal_llmj.py', '--mode', 'bench',
                                 '--budget', '1', '--spent', '0'], cwd=ROOT,
                                stdout=log, stderr=subprocess.STDOUT)
        try:
            code = proc.wait(timeout=660)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()
            code = proc.wait(timeout=20)
    text = (DEST/'modal_bench.log').read_text(encoding='utf-8')
    appids = list(dict.fromkeys(re.findall(r'\bap-[A-Za-z0-9]+', text)))
    stops = []
    for appid in appids:
        stopped = subprocess.run([MODAL, 'app', 'stop', appid, '--yes'],
                                 capture_output=True, text=True, encoding='utf-8', timeout=40)
        stops.append(dict(app_id=appid, exit=stopped.returncode,
                          stdout=stopped.stdout, stderr=stopped.stderr))
    wall = time.monotonic()-start
    report = dict(exit=code, timed_out=timed_out, client_wall_s=wall,
                  rate_upper_usd_s=RATE, all_wall_h100_upper_usd=wall*RATE,
                  reserve_usd=1-wall*RATE, app_ids=appids, stops=stops,
                  note='Conservative bound, not billing: includes CPU-only image build at H100 rate')
    audit.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    sys.exit(code if code else int(timed_out))


if __name__ == '__main__':
    main()
