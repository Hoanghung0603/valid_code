"""merge_list.py <cands.pkl> <extra.pkl>: add the lists in extra.pkl ({qid: {name: int32 chunk indices}}) to cands.pkl."""
import pickle, sys
c = pickle.load(open(sys.argv[1], "rb"))
x = pickle.load(open(sys.argv[2], "rb"))
for q, L in x.items():
    c["lists"][q].update(L)
pickle.dump(c, open(sys.argv[1], "wb"))
print("merged", sorted(next(iter(x.values()))), "into", len(x), "queries")
