"""Read-only streaming hashes of original vi37/vi36 source artifacts; writes only vi41."""
import hashlib
import json
from pathlib import Path

root=Path('D:/r2ai/data/sub')
names=['vi37/submission_vi37.zip','vi37/q8b_v37.json','vi37/cands_v37.json.zst',
       'diag37/docwin_max_v37.json','vi36/in1/pool_v36.json','vi36/in1/dense_v36.json',
       'vi36/in1/hyde_v36.json','vi36/in1/ref_chunks_v36.json']
result={}
for name in names:
    p=root/name
    sha=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4<<20),b''):
            sha.update(block)
    result[name]={'bytes':p.stat().st_size,'sha256':sha.hexdigest()}
target=root/'vi41/original_inputs_sha.json'
if target.exists():
    assert json.loads(target.read_text())==result,'Immutable source input hashes changed'
    (root/'vi41/original_inputs_final_check.json').write_text(json.dumps({'files':len(result),'all_unchanged':True}))
else:
    target.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
