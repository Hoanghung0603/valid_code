"""Verify only the named vi55 mirror files, no corpus/crawl access."""
import hashlib
import json
from pathlib import Path
root=Path('D:/r2ai/data/sub/vi55')
manifest=json.loads((root/'recovery_mirror_manifest.json').read_text(encoding='utf-8'))
for name,meta in manifest.items():
    assert Path(name).name==name
    p=root/name;h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    assert p.stat().st_size==meta['bytes'] and h.hexdigest()==meta['sha256'],name
report=json.loads((root/'phase1_report.json').read_text(encoding='utf-8'))
assert report['status']=='DONE' and report['total_upper_usd']<=26 and not report['phase2']
assert report['top1000_audit']['queries']==1200 and report['top1000_audit']['documents']==729704
print('VI55_MIRROR_VERIFIED files='+str(len(manifest)),flush=True)
