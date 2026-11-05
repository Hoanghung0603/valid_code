"""Local compressed-input provenance; never launches anything."""
import hashlib
import json
from pathlib import Path
import zstandard as zstd

T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi45')
m=json.loads((T/'prepare_vi44_report.json').read_text())
seen=set()
for name,meta in m['files'].items():
    p=T/'modal_input'/name
    assert p.stat().st_size==meta['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
    count=0
    with zstd.open(p,'rt',encoding='utf-8') as f:
        for line in f:
            r=json.loads(line)
            ident=hashlib.sha256((r['qid']+'\0'+str(r['doc_id'])+'\0'+r['text']).encode()).hexdigest()
            assert r['cid']==ident and ident not in seen
            seen.add(ident)
            count+=1
    assert count==m['counts'][name]
assert len(seen)==m['scored_candidates']
report=dict(files=len(m['files']),pairs=len(seen),sha_count_cid_duplicate_bad=0)
(T/'input_audit.json').write_text(json.dumps(report))
print(json.dumps(report))
