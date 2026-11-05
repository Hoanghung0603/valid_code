"""Lọc d{1,2,3}_new.json (output kaggle_newdocs, có thể vài GB vì kèm cửa sổ văn bản) theo kiểu streaming (ijson):
chỉ giữ doc mới có điểm > điểm hạng 300 của list cũ (thr.json) – các doc khác không thể lọt vào top-300 sau khi gộp.
    python scripts/newdocs_filter.py --kout data/cache/newdocs_out_ph --thr data/cache/kg_newdocs_ph/thr.json
Ghi đè tại chỗ (qua file .tmp)."""
import argparse, json, os
import ijson

ap = argparse.ArgumentParser(); ap.add_argument("--kout", required=True); ap.add_argument("--thr", required=True); a = ap.parse_args()
thr = json.load(open(a.thr))
for i, k in enumerate(("d1", "d2", "d3")):
    f = f"{a.kout}/{k}_new.json"; kept = tot = 0
    with open(f, "rb") as fi, open(f + ".tmp", "w") as fo:
        fo.write("{"); first = True
        for q, lst in ijson.kvitems(fi, "", use_float=True):
            t = thr.get(q, [-1, -1, -1])[i]; keep = [x for x in lst if x[0] > t]; tot += len(lst); kept += len(keep)
            fo.write(("" if first else ", ") + json.dumps(q) + ": " + json.dumps(keep, ensure_ascii=False)); first = False
        fo.write("}")
    os.replace(f + ".tmp", f); print(k, "giữ", kept, "/", tot, "->", os.path.getsize(f) // 2**20, "MB", flush=True)
