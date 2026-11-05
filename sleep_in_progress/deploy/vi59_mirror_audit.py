"""Verify own mirrored deliverables on b660; no historical files modified."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path('D:/r2ai/data/sub/vi59')


def fingerprint(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return dict(bytes=p.stat().st_size,sha256=h.hexdigest())


def main():
    manifest=json.loads((ROOT/'mirror_manifest.json').read_text(encoding='utf-8'))
    for rel,meta in manifest.items():assert fingerprint(ROOT/rel)==meta,(rel,meta)
    cx=ROOT/'cx_extra'
    if not (cx/'MANIFEST.json').exists():
        with zipfile.ZipFile(ROOT/'cx_extra.zip') as z:z.extractall(cx)
    for rel,meta in json.loads((cx/'MANIFEST.json').read_text(encoding='utf-8'))['files'].items():
        assert fingerprint(cx/rel)==meta,rel
    report=dict(status='MIRROR_SHA_PASS',files=len(manifest),cx_files=len(list(cx.rglob('*'))))
    (ROOT/'mirror_verified.json').write_text(json.dumps(report),encoding='utf-8')
    print(json.dumps(report))


if __name__=='__main__':main()
