"""Read-only named vi54 backup hashes (small control job)."""
import argparse
import hashlib
import json
from pathlib import Path

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=Path('D:/r2ai/data/sub/vi54'))
    a=ap.parse_args();manifest=json.loads((a.root/'backup_manifest.json').read_text(encoding='utf-8'))
    for name,meta in manifest.items():
        p=a.root/name;h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        assert p.stat().st_size==meta['bytes'] and h.hexdigest()==meta['sha256'],name
        print('BACKUP_SHA_PASS '+name,flush=True)
    print('VI54_BACKUP_VERIFIED',flush=True)
