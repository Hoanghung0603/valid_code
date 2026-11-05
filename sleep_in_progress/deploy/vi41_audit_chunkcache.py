"""CPU read-only overlap inventory; no changes to old candidate caches."""
import json
from pathlib import Path
import sys
sys.path.insert(0,'D:/r2ai/scripts')
import dwmax_diag as dw

root=Path('D:/r2ai/data/sub')
old=dw.read(root/'vi37/cands_v37.json.zst')
new=dw.read(root/'vi41/new_cands_u900.json.zst')
overlap=sum(len({str(r[0]) for r in rs}&{str(r[0]) for r in old.get(q,[])}) for q,rs in new.items())
print(json.dumps({'new_cached_pairs_overlapping_old_chunk_docs':overlap,
                  'policy':'old candidates win for overlapping doc; same direct chunk behavior'}))
