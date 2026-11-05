"""CPU source/reranker distributions for every new candidate, no quality gate."""
import hashlib
import json
import numpy as np
from vi41_local import TEMP

fresh=json.loads((TEMP/'new_ids.json').read_text())
why=json.loads((TEMP/'new_provenance.json').read_text())
rr=json.loads((TEMP/'rr_new.json').read_text())
lengths=json.loads((TEMP/'export_lengths.json').read_text())
source={s:[] for s in ('dense','dchunk','bm25','neardup')}
empty={s:0 for s in source}
for q,rows in rr.items():
    assert set(fresh[q])=={r[0] for r in rows}
    for d,i,score in rows:
        for s in why[q][str(d)]:
            if lengths[str(d)]['windows']:
                source[s].append(score)
            else:
                empty[s]+=1
result={s:dict(n=len(v),per_query=len(v)/1200,mean=float(np.mean(v)) if v else None,
               empty_excluded=empty[s],
               quantiles=np.quantile(v,[0,.1,.5,.9,1]).tolist() if v else [],
               above_minus2=sum(x>-2 for x in v),above_zero=sum(x>0 for x in v)) for s,v in source.items()}
(TEMP/'candidate_logit_report.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
