"""Offline retrieval provenance diversity, no label/quality gate."""
from collections import Counter
import json
import numpy as np
from vi41_local import TEMP

provenance=json.loads((TEMP/'new_provenance.json').read_text())
sources=('dense','dchunk','bm25','neardup')
patterns=Counter()
matrix={a:{b:0 for b in sources} for a in sources}
counts=[]
for q,ds in provenance.items():
    counts.append(len(ds))
    for labels in ds.values():
        patterns['+'.join(sorted(labels))]+=1
        for a in labels:
            for b in labels:
                matrix[a][b]+=1
result=dict(new_pairs=sum(counts),per_query_quantiles=np.quantile(counts,[0,.1,.5,.9,1]).tolist(),
    mean_per_query=float(np.mean(counts)),provenance_patterns=dict(patterns),
    pairwise_overlap_counts=matrix,method='counts only; no relevance/P or quality threshold')
(TEMP/'source_overlap_report.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
