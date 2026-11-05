"""b660 short mirror extraction, SHA validation; no GPU/index/old-data actions."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main(root,manifest):
    report=json.loads(manifest.read_text());dest=root/'artifacts';dest.mkdir(exist_ok=True)
    for part in report['parts']:
        p=root/part['name'];assert p.stat().st_size==part['bytes'] and sha(p)==part['sha256']
        with zipfile.ZipFile(p) as z:
            assert all(not n.startswith('/') and '..' not in Path(n).parts for n in z.namelist())
            z.extractall(dest)
        for n,r in part['files'].items():
            p=dest/n;assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    (root/f"mirror_{report['tier']}_verified.json").write_text(json.dumps(dict(parts=len(report['parts']),sha_pass=True)),encoding='utf-8')
    print('MIRROR_VERIFIED',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--manifest',type=Path,required=True)
    a=ap.parse_args();main(a.root,a.manifest)
