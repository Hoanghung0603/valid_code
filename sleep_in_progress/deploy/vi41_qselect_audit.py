"""Audit selected 8B pairs against new-pool provenance and nonempty exported text."""
import json
from vi41_local import TEMP

cands=json.loads((TEMP/'modal_input/q8b_cands.json').read_text())
new=json.loads((TEMP/'new_ids.json').read_text())
lengths=json.loads((TEMP/'export_lengths.json').read_text())
why=json.loads((TEMP/'new_provenance.json').read_text())
selected=0
sources={s:0 for s in ('dense','dchunk','bm25','neardup')}
counts=[]
for q in map(str,range(1,1201)):
    rows=cands.get(q,[])
    counts.append(len(rows))
    assert len(rows)==len({r[0] for r in rows})
    for d,i in rows:
        assert d in new[q] and lengths[str(d)]['windows']>0
        assert lengths[str(d)]['idx']==i
        selected+=1
        for s in why[q][str(d)]:
            sources[s]+=1
assert selected==json.loads((TEMP/'qselect_report.json').read_text())['pairs']==22682
result=dict(pairs=selected,queries_with_pairs=sum(n>0 for n in counts),
    mean_per_query=selected/1200,max_per_query=max(counts),by_source=sources,
    exact_new_ids_nonempty_idx=True,duplicates=0)
(TEMP/'qselect_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
