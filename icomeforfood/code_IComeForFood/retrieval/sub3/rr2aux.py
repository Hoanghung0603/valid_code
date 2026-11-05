"""vnrr.py / q8rr.py output (chunk scores) -> doc scores in prf.py's layout {0: {qid: {doc: score}}}, for make_sub3 --prf ... --prf-k 0.
    python rr2aux.py <q8rr.pkl> <out.pkl>      (chunk files under $R2AI_DATA, default D:/r2ai_data)"""
import os, pickle, sys
from pathlib import Path
import pyarrow.parquet as pq
DATA = Path(os.environ.get("R2AI_DATA", "D:/r2ai_data"))
files = [f for d in [DATA / f"chunks_{s}/s400" for s in ("sub2", "sub3add", "sub4add", "sub5add")]
         for f in sorted(Path(d).glob("*.parquet"))]
doc = pq.read_table(files, columns=["doc_id"]).column("doc_id").to_numpy()
v = pickle.load(open(sys.argv[1], "rb"))
out = {0: {q: {int(doc[i]): s for i, s in r["vi"].items()} for q, r in v.items()}}
pickle.dump(out, open(sys.argv[2], "wb"))
print(len(out[0]), sum(len(x) for x in out[0].values()) / len(out[0]))
