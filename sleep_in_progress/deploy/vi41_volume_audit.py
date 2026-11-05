"""Download own Volume inputs to separate staging and compare full SHA, then remove exact files."""
import hashlib
import json
from vi41_local import TEMP,cmd

report=json.loads((TEMP/'transfer_manifest.json').read_text())
directory=TEMP/'volume_verify'
directory.mkdir(exist_ok=True)
checked=[]
for row in report['files']:
    p=directory/row['name']
    cmd('volume','get','--force','r2ai-runner','/in/vi41/'+row['name'],str(p))
    assert p.stat().st_size==row['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],row['name']
    checked.append(row)
    p.unlink()
result=dict(files=len(checked),bytes=sum(r['bytes'] for r in checked),remote_sha_ok=True,
            staging_removed_exact_names=True)
(TEMP/'volume_sha_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
