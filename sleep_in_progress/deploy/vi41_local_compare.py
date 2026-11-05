"""Independent local doc/chunk delta and exact by-ID comparison against immutable vi37 ZIP."""
import json
import hashlib
import zipfile
from vi41_local import TEMP

oldpath=TEMP.parent/'vi37/submission_vi37.zip'
assert hashlib.sha256(oldpath.read_bytes()).hexdigest()=='95cfd48cf70bbc7dc04c83757cbbca3d179205edff0815aa689d651679b4086a'
with zipfile.ZipFile(oldpath) as z:
    old={str(r['id']):r for r in json.loads(z.read(z.namelist()[0]))}
with zipfile.ZipFile(TEMP/'submission_vi41.zip') as z:
    new={str(r['id']):r for r in json.loads(z.read('submission.json'))}
assert set(old)==set(new)=={str(i) for i in range(1,1201)}
a=b=0;j=[];doc_equal=chunk_equal=0
chunk_in=chunk_out=0
for q,row in new.items():
    od,nd=set(old[q]['relevant_docs']),set(row['relevant_docs'])
    a+=len(nd-od);b+=len(od-nd);j.append(len(od&nd)/len(od|nd))
    doc_equal+=old[q]['relevant_docs']==row['relevant_docs']
    oc={(c['doc_id'],c['chunk_text']) for c in old[q]['relevant_chunks']}
    nc={(c['doc_id'],c['chunk_text']) for c in row['relevant_chunks']}
    chunk_in+=len(nc-oc);chunk_out+=len(oc-nc)
    chunk_equal+=old[q]['relevant_chunks']==row['relevant_chunks']
result=dict(queries=1200,docs_in_per_query=a/1200,docs_out_per_query=b/1200,jaccard_mean=sum(j)/1200,
    doclist_identical=doc_equal,chunklist_identical=chunk_equal,
    chunks_in_per_query=chunk_in/1200,chunks_out_per_query=chunk_out/1200)
reported=json.loads((TEMP/'stats_report.json').read_text())
for k in ('docs_in_per_query','docs_out_per_query','jaccard_mean'):
    assert abs(result[k]-reported[k])<1e-10,(k,result[k],reported[k])
(TEMP/'local_compare.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
