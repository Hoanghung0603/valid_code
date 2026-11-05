"""One-shot CPU input hash verification before starting build Task Scheduler."""
import hashlib
import json
from pathlib import Path

out=Path('D:/r2ai/data/sub/vi41')
assert not Path('D:/r2ai/locks/heavy_job.lock').exists(),'Another CPU owner; wait, do not touch'
manifest=json.loads((out/'score_transfer_manifest.json').read_text())
assert {r['name'] for r in manifest['files']}=={'rr_new.json','q8b_new.json','docwin_new.json.zst'}
for row in manifest['files']:
    p=out/'modal_out'/row['name']
    assert p.stat().st_size==row['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p
for name in ('sanity_report.json','sanity_chain_report.json'):
    assert json.loads((out/name).read_text())['doclist_equal']==1200
assert json.loads((out/'sanity_chain_report.json').read_text())['chunks_equal']==1200
assert json.loads((out/'input_audit_report.json').read_text())['source_caps_ok']
assert json.loads((out/'pre_chunks_report.json').read_text())['verbatim_bad']==0
print(json.dumps({'score_files_sha_ok':len(manifest['files']),'sanity_doc_chunk':1200,'lock':False}))
