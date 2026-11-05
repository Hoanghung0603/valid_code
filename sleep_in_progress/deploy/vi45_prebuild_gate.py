"""b660: require complete SHA-verified transferred scores before enabling build."""
import hashlib
import json
from pathlib import Path

O=Path('D:/r2ai/data/sub/vi45')
m=json.loads((O/'score_transfer_manifest.json').read_text())
actual={}
count=0
for p in (O/'scores').glob('*.logits.json'):
    actual[p.name]=dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
    count+=len(json.loads(p.read_text()))
assert actual==m['files']
assert len(actual)==20 and count==317601
pre=json.loads((O/'preverify_report.json').read_text())
assert pre['verbatim_bad']==0 and pre['candidates']==356314
assert pre['candidate_fingerprint']['sha256']==hashlib.sha256((O/'candidates_vi45_vi44.json.zst').read_bytes()).hexdigest()
print(json.dumps(dict(files=len(actual),pairs=count,sha_bad=0,ready=True)))
