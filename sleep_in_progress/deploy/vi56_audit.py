"""Independent local immutable-input audit; no GPU/network/submission."""
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vi56 import read,write,iterz,fingerprint,TEMP
from vi45 import cid

out=TEMP/'vi56';prep=read(out/'prepare_report.json');counts=Counter();seen=set();parts={}
fresh=read(out/'new_ids.json');raw=read(TEMP/'vi55/q3e8b_full_top1000.json.zst')
ctx=read(TEMP/'vi50/fusion_context.json.zst');oldq=read(TEMP/'vi51/cosines_vi51b.json')
bad=set(map(int,read(TEMP/'vi45/bad_ids.json')))-set(map(int,read(TEMP/'vi45/new_ids.json')))
for q,ds in fresh.items():
    assert not set(ds)&bad
    assert not set(ds)&{int(r[0]) for r in ctx['data']['pool'][q]}
    assert set(ds)<={int(d) for d,s in raw[q][:100]}
    assert not set(map(str,ds))&set(oldq[q])
for name,meta in prep['parts'].items():
    p=out/'modal_input'/name;assert fingerprint(p)=={k:meta[k] for k in ('bytes','sha256')}
    rows=list(iterz(p));assert len(rows)==meta['count'];parts[name]=len(rows)
    for r in rows:
        assert r['identity'] not in seen;seen.add(r['identity']);counts[r['kind']]+=1
        if r['kind']=='d8w':assert cid(r['qid'],dict(doc_id=r['doc_id'],chunk_text=r['text']))==r['identity']
expected={r[0] for stage in ('bge','qwen') for r in read(out/f'expected_{stage}.json')}
assert expected==seen and dict(counts)==prep['counts']
write(out/'input_audit.json',dict(parts=parts,counts=dict(counts),raw_pairs=prep['pairs'],eligible_pairs=sum(map(len,fresh.values())),
    exact_ids=True,all_sha=True,no_duplicate_identity=True,all_d8w_cid=True,top100_only=True,excluded_bad=len(read(out/'selection_report.json')['contaminated_excluded']),
    old_inputs={str(p):fingerprint(p) for p in (TEMP/'vi50/fusion_context.json.zst',TEMP/'vi50/d8w.json',TEMP/'vi51/cosines_vi51b.json',TEMP/'vi48/submission_vi48.zip',TEMP/'vi53/submission_vi53.zip')}))
print('INPUT_AUDIT_PASS',len(seen),dict(counts),flush=True)
