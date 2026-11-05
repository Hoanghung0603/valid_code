"""vnrr.py output (chunk scores) -> doc scores in prf.py's layout {0: {qid: {doc: score}}}, for make_sub3 --prf ... --prf-k 0.
    python vn2aux.py <vnrr.pkl> <out.pkl>"""
import pickle, sys
from pathlib import Path
import pyarrow.parquet as pq
files = [f for d in ["D:/r2ai_data/chunks_sub2/s400", "D:/r2ai_data/chunks_sub3add/s400", "D:/r2ai_data/chunks_sub4add/s400"]
         for f in sorted(Path(d).glob("*.parquet"))]
doc = pq.read_table(files, columns=["doc_id"]).column("doc_id").to_numpy()
v = pickle.load(open(sys.argv[1], "rb"))
out = {0: {q: {int(doc[i]): s for i, s in r["vi"].items()} for q, r in v.items()}}
pickle.dump(out, open(sys.argv[2], "wb"))
print(len(out[0]), sum(len(x) for x in out[0].values()) / len(out[0]))
