"""Independent local cache/input/vi44 relation check before final assembly."""
import hashlib
import json
from pathlib import Path
import sys

import zstandard as zstd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi41c import load_zip
from dwmax_diag import read
import vi45

T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi45')
cache=read(T/'candidates_vi45_vi44.json.zst')
ref=load_zip(T.parent/'vi44/submission_vi44.zip')
m=read(T/'prepare_vi44_report.json')
inputs={str(r['id']):set() for r in ref}
for name in m['files']:
    with zstd.open(T/'modal_input'/name,'rt',encoding='utf-8') as f:
        for r in map(json.loads,f):
            inputs[r['qid']].add(r['cid'])
changed=orderok=full=scoredzero_changed=scoredzero_order=0
for it in ref:
    q=str(it['id'])
    rows=cache[q]
    ids={c['cid'] for c in rows}
    assert len(rows)==len(ids)
    assert all(c['cid']==vi45.cid(q,c) for c in rows)
    assert all(abs(c['s']-c['zc']-4*c['zd'])<1e-10 for c in rows)
    assert {str(c['doc_id']) for c in rows}<=set(map(str,it['relevant_docs']))
    assert inputs[q]<=ids
    assert {vi45.key(c) for c in it['relevant_chunks']}<={vi45.key(c) for c in rows}
    before=[vi45.key(c) for c in it['relevant_chunks']]
    after=[vi45.key(c) for c in vi45.rank(rows)[:len(before)]]
    orderok+=before==after
    changed+=set(before)!=set(after)
    zrank=[vi45.key(c) for c in vi45.rank(rows,{ident:0. for ident in inputs[q]})[:len(before)]]
    scoredzero_changed+=set(before)!=set(zrank)
    scoredzero_order+=before==zrank
    full+=len(rows)
assert changed==m['sanity_changed_sets'] and orderok==m['sanity_zero_order']
assert full==m['full_candidates']
result=dict(queries=1200,full=full,scored=sum(map(len,inputs.values())),changed_sets=changed,
    order_equal=orderok,scored_zero_changed_sets=scoredzero_changed,scored_zero_order_equal=scoredzero_order,
    valid_doc_formula_cid_duplicates_missing_vi44_bad=0,
    cache_sha256=hashlib.sha256((T/'candidates_vi45_vi44.json.zst').read_bytes()).hexdigest())
(T/'candidate_audit.json').write_text(json.dumps(result))
print(json.dumps(result))
