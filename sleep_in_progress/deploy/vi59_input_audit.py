"""CPU-only, exact identity/type/SHA audit of extra and historical reuse."""
from collections import Counter
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,iterz,fingerprint
OUT=ROOT.parent/'r2ai_deliver/cand/vi59d'


def main():
    prep=read(OUT/'extra/prepare_report.json')
    old_ids=set()
    for p in sorted((OUT/'inputs/vi56/modal_input').glob('*.jsonl.zst')):
        old_ids.update(r['identity'] for r in iterz(p))
    expected={stage:{r[0]:(str(r[1]),int(r[2]),r[3]) for r in read(OUT/f'extra/expected_{stage}.json')}
              for stage in ('qwen','bge')}
    seen=set();counts=Counter();parts={}
    empty={(str(q),int(d)) for q,d in prep['empty_pairs']}
    fresh=read(OUT/'extra/new_ids.json')
    for name,meta in prep['parts'].items():
        p=OUT/'extra/modal_input'/name
        assert fingerprint(p)=={k:meta[k] for k in ('bytes','sha256')}
        rows=list(iterz(p));assert len(rows)==meta['count']
        for r in rows:
            ident=r['identity'];q=str(r['qid']);d=int(r['doc_id'])
            assert ident not in seen and ident not in old_ids
            assert expected[meta['stage']][ident]==(q,d,r['kind'])
            assert d in fresh[q] and (q,d) not in empty
            assert r['query'].strip() and r['text'].strip()
            seen.add(ident);counts[r['kind']]+=1
        parts[name]=fingerprint(p)
    assert seen==set(expected['bge'])|set(expected['qwen'])
    assert dict(counts)==prep['counts']
    result=dict(status='PASS',pairs=len(seen),counts=dict(counts),parts=parts,
                old_new_identity_overlap=0,exact_expected=True)
    write(OUT/'extra_input_audit.json',result)
    print({k:v for k,v in result.items() if k!='parts'},flush=True)


if __name__=='__main__':main()
