"""Independent vi47 document-protection reconstruction after build."""
import gc
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi41c import load_zip, indexed, digest
T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))
O=T/'vi47'
report=json.loads((O/'build_report_vi47.json').read_text())
zp=O/'submission_vi47.zip'
assert hashlib.sha256(zp.read_bytes()).hexdigest()==report['sha256']
actual=indexed(load_zip(zp))
lengths={q:len(r['relevant_docs']) for q,r in actual.items()}
chunks={q:digest(r['relevant_chunks']) for q,r in actual.items()}
protected={q:{str(c['doc_id']) for c in r['relevant_chunks']} for q,r in actual.items()}
docs={q:r['relevant_docs'] for q,r in actual.items()}
del actual
gc.collect()
ref={q:r['relevant_docs'] for q,r in indexed(load_zip(T/'vi37/submission_vi37.zip')).items()}
assert all(lengths[q]==len(ref[q]) for q in ref)
candidate={q:r['relevant_docs'] for q,r in indexed(load_zip(T/'vi41c/submission_vi41c.zip')).items()}
missing_total=affected=0
for q in ref:
    available=set(map(str,candidate[q]))
    missing=[d for d in ref[q] if str(d) in protected[q] and str(d) not in available]
    assert protected[q]<=set(map(str,ref[q]))
    removable=[d for d in reversed(candidate[q]) if str(d) not in protected[q]]
    assert len(removable)>=len(missing)
    drop=set(map(str,removable[:len(missing)]))
    expected=[d for d in candidate[q] if str(d) not in drop]+missing
    assert expected==docs[q],q
    missing_total+=len(missing)
    affected+=bool(missing)
del docs,ref,candidate
gc.collect()
source=indexed(load_zip(T/'vi45/submission_vi45.zip'))
assert set(source)==set(chunks)==set(range(1,1201))
assert all(digest(r['relevant_chunks'])==chunks[q] for q,r in source.items())
assert missing_total==report['protected_missing_total'] and affected==report['protected_affected_queries']
result=dict(independent_doc_rule_equal=1200,chunks_equal_vi45=1200,doc_counts_equal_vi37=1200,
            missing_added=missing_total,affected_queries=affected,sha256=report['sha256'],zip_bytes=zp.stat().st_size)
(O/'independent_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
