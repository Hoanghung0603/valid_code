"""Unpack only transferred vi38 job outputs and verify each local/b660 hash."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

ap = argparse.ArgumentParser()
ap.add_argument('mode', choices=['bulk', 'confirm'])
a = ap.parse_args()
base = Path('D:/r2ai/data/sub/vi38')
dest = base/(a.mode+'_out')
dest.mkdir(exist_ok=True)
with zipfile.ZipFile(base/(a.mode+'_output_transfer.zip')) as z:
    assert all(Path(n).name == n for n in z.namelist())
    z.extractall(dest)
verification = json.loads((dest/'transfer_verification.json').read_text(encoding='utf-8'))
for name, expected in verification.items():
    p = dest/name
    assert p.stat().st_size == expected['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected['sha256']
print('TRANSFER SHA PASS', len(verification), flush=True)
