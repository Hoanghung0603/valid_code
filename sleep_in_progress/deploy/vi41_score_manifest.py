"""Local score-output manifest for exact b660 upload verification."""
import hashlib
import json
from vi41_local import TEMP

names=['rr_new.json','docwin_new.json.zst','q8b_new.json']
for stage in ('rr','docwin','q8b'):
    a=json.loads((TEMP/(stage+'_audit.json')).read_text())
    assert a['finite']
report={'files':[dict(name=n,bytes=(TEMP/n).stat().st_size,
    sha256=hashlib.sha256((TEMP/n).read_bytes()).hexdigest()) for n in names]}
(TEMP/'score_transfer_manifest.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
