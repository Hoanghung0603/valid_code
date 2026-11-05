"""Download Modal output directory, verify ready-part SHA, prepare b660 transfer."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import zipfile

BASE = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38')
MODAL = __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=['bulk', 'confirm'])
    a = ap.parse_args()
    os.environ.update(PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
    dest = BASE/(a.mode+'_out')
    if not (dest/f'report_{a.mode}.json').exists():
        subprocess.run([MODAL, 'volume', 'get', 'r2ai-runner', '/out/vi38/'+a.mode+'_out', str(dest), '--force'], check=True)
    nested = dest/(a.mode+'_out')
    if nested.exists():
        for p in nested.iterdir():
            assert p.is_file() and not (dest/p.name).exists()
            p.rename(dest/p.name)
        nested.rmdir()
    src = BASE/(a.mode+'_input')
    manifest = json.loads((src/'manifest.json').read_text(encoding='utf-8'))
    total, labels = 0, {'A': {}, 'B': {}}
    seen = set()
    for name in manifest['modes'][a.mode]:
        ob = json.loads((dest/(name+'.out.json')).read_text(encoding='utf-8'))
        assert ob['input_sha256'] == manifest['files'][name]['sha256']
        total += len(ob['rows'])
        for row in ob['rows']:
            assert row['pair_id'] not in seen
            seen.add(row['pair_id'])
            assert row['status'] == 'ok' and all(math.isfinite(row[f'logp{i}']) for i in range(3))
            role, label = row['role'], str(row['label'])
            labels[role][label] = labels[role].get(label, 0)+1
    if a.mode == 'bulk':
        assert total == len(seen) == 99675
    files = [p for p in dest.iterdir() if p.is_file()]
    verification = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    (dest/'transfer_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
    with zipfile.ZipFile(BASE/(a.mode+'_output_transfer.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in files+[dest/'transfer_verification.json']:
            z.write(p, p.name)
    print(json.dumps(dict(pairs=total, labels=labels, files=len(files),
                         transfer_bytes=(BASE/(a.mode+'_output_transfer.zip')).stat().st_size)), flush=True)

if __name__ == '__main__':
    main()
