"""Audit downloaded scores and produce exact one-copy upload inventory."""
import hashlib
import json
import math
from pathlib import Path
import zstandard as zstd

T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi45')
m=json.loads((T/'prepare_vi44_report.json').read_text())
expected={}
for name in m['files']:
    with zstd.open(T/'modal_input'/name,'rt',encoding='utf-8') as f:
        expected[name]={r['cid']:(r['qid'],str(r['doc_id'])) for r in map(json.loads,f)}
files=list((T/'outputs').glob('*/*/*.logits.json'))
actual={}
inventory=[]
for p in files:
    name=p.name.replace('.logits.json','.jsonl.zst')
    assert name not in actual
    rows=json.loads(p.read_text())
    actual[name]={r['cid']:(r['qid'],str(r['doc_id'])) for r in rows}
    assert len(actual[name])==len(rows)
    assert actual[name]==expected[name]
    assert all(math.isfinite(r['logit8c']) for r in rows)
    inventory.append(dict(path=str(p),name=p.name,bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest(),pairs=len(rows)))
missing=sorted(set(expected)-set(actual))
report=dict(files=len(files),pairs=sum(map(len,actual.values())),missing=missing,inventory=inventory)
(T/'scores_audit.json').write_text(json.dumps(report,indent=2))
(T/'score_transfer_manifest.json').write_text(json.dumps(dict(files={r['name']:dict(bytes=r['bytes'],sha256=r['sha256']) for r in inventory}),indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='inventory'}))
assert not missing, 'Cannot build with missing parts'
