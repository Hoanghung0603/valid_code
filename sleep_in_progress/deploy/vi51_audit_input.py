"""Independent local fixed-input audit; no models or mutation of source artifacts."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi48 import fingerprint,read,write
from vi51 import iterz
import dwmax_diag as dw

root=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi51')
prep=read(root/'prepare_report.json');ctx=dw.read(root/'fusion_context.json.zst')
seen=set();total=0;parts=[]
pool=read(root/'modal_input/pool.json')
assert pool=={q:[int(h[0]) for h in rows] for q,rows in ctx['data']['pool'].items()}
for n,meta in prep['parts'].items():
    p=root/'modal_input'/n
    assert fingerprint(p)=={k:meta[k] for k in ('bytes','sha256')}
    count=tokens=0
    for r in iterz(p):
        assert r['id'] not in seen and 1<=len(r['input_ids'])<=2048
        seen.add(r['id']);count+=1;tokens+=len(r['input_ids'])
        assert all(isinstance(t,int) and 0<=t<151700 for t in r['input_ids'])
    assert count==meta['docs'] and tokens==meta['tokens']
    total+=tokens;parts.append(n)
assert seen=={d for ds in pool.values() for d in ds}
assert len(seen)==prep['documents'] and total==prep['tokens']
assert sum(map(len,pool.values()))==prep['pairs']
queries=read(root/'modal_input/queries.json')
assert len(queries)==1200 and {r['id'] for r in queries}==set(range(1,1201))
assert all(1<=len(r['input_ids'])<=256 for r in queries)
write(root/'input_audit.json',dict(documents=len(seen),pairs=prep['pairs'],tokens=total,parts=len(parts),
    sha_all_pass=True,doc_dedup=True,pool_exact=True,queries=1200,token_caps=True))
print((root/'input_audit.json').read_text())
