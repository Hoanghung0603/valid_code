"""Extract vi53 doclists only and reusable target vectors; no chunks changed."""
import json
from pathlib import Path
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import write
TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))
out=TEMP/'vi55'
with zipfile.ZipFile(TEMP/'vi53/submission_vi53.zip') as z:
    with z.open(z.namelist()[0]) as f:
        data=json.load(f)
doclists={str(r['id']):list(map(int,r['relevant_docs'])) for r in data}
del data
write(out/'vi53_doclists.json',doclists)
wanted={d for ds in doclists.values() for d in ds};vectors={}
for p in (TEMP/'vi51b/outputs').rglob('*.emb.npz'):
    z=np.load(p)
    emb=z['emb']
    for i,d in enumerate(z['ids']):
        if int(d) in wanted:vectors[int(d)]=emb[i].copy()
assert set(vectors)==wanted
ids=sorted(vectors)
np.savez(out/'vi53_target_embeddings.npz',ids=np.asarray(ids,dtype=np.int64),emb=np.asarray([vectors[d] for d in ids],dtype=np.float16))
print('TARGETS '+str(len(ids)),flush=True)
