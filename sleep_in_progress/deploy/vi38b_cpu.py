"""Scheduled CPU stages, own-lock wait (5min, <=90min), explicit output pack."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path('D:/r2ai')
BASE = ROOT/'data/sub/vi38'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['prepare', 'finish'])
    a = ap.parse_args()
    lock = ROOT/'locks/heavy_job.lock'
    started = time.monotonic()
    def wait_lock():
        while lock.exists():
            print('WAIT foreign lock', lock.read_text(encoding='utf-8'), flush=True)
            if time.monotonic()-started >= 5400:
                raise RuntimeError('STOP foreign lock still held after 90min')
            time.sleep(300)
    stages = ['merge', 'confirm'] if a.mode == 'prepare' else ['cmerge', 'swap', 'build', 'stats']
    for stage in stages:
        wait_lock()
        subprocess.run([sys.executable, str(ROOT/'scripts/vi38.py'), stage, '--rule', 'b'], check=True)
    if a.mode == 'prepare':
        manifest = json.loads((BASE/'export/manifest.json').read_text(encoding='utf-8'))
        names = ['manifest.json', *manifest['modes']['confirm']]
        with zipfile.ZipFile(BASE/'confirm_input.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for n in names:
                z.write(BASE/'export'/n, n)
        print('CONFIRM PACKAGE', (BASE/'confirm_input.zip').stat().st_size, flush=True)
    else:
        # gate2 itself acquires the same owned lock; no nesting.
        wait_lock()
        subprocess.run([sys.executable, str(ROOT/'scripts/vi38_gate2.py')], check=True)
    print('CPU DONE', a.mode, flush=True)

if __name__ == '__main__':
    main()
