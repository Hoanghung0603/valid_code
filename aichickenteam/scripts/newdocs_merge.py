"""Gộp output kernel kaggle_newdocs/ (đặc trưng cho doc mới crawl) vào các file cache của pipeline → bản *_v3.

    python scripts/newdocs_merge.py --kout data/cache/newdocs_out --docs <dir có new_docs.parquet> --tag 91000
Sau đó chạy ltr.py / fuse_dense.py với các đường dẫn *_v3 (in ra cuối).
"""
import argparse
import json
import os
import shutil

import numpy as np
import pandas as pd

def paths(v):   # v=2: bản gốc trước khi gộp doc mới; v>=3: sau mỗi lần gộp
    if v == 2:
        return {"d1": "data/cache/dense_gn_top300_v2.json", "d2": "data/cache/kaggle_bgem3_out/dense_chunks_top.json",
                "d3": "data/cache/dense_chunks_top_v2.json", "e5": "data/cache/e5", "rr": lambda n: f"data/cache/{n}.parquet"}
    return {"d1": f"data/cache/dense_gn_top300_v{v}.json", "d2": f"data/cache/dense_bgem3_chunks_v{v}.json",
            "d3": f"data/cache/dense_chunks_top_v{v}.json", "e5": f"data/cache/e5_v{v}", "rr": lambda n: f"data/cache/{n}_v{v}.parquet"}
RR = {"rr": "rr_bge_all", "rr2": "rr_win", "rr3": "ce_ft", "rr4": "ce_ftwin", "gnft": "gnft"}
if os.environ.get("R2AI_MERGE_RR5"):   # ce2 (v62 dùng ce2b_v8) – cột rr5 từ kernel (model ce2_all)
    RR["rr5"] = os.environ["R2AI_MERGE_RR5"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ver", type=int, default=3, help="phiên bản tạo ra (gộp lên bản ver-1)")
    ap.add_argument("--kout", required=True); ap.add_argument("--docs", required=True); ap.add_argument("--tag", default="91000")
    a = ap.parse_args()
    K = a.kout; OLD, NEW = paths(a.ver - 1), paths(a.ver)
    RM = a.ver - 1 >= 3 and not os.environ.get("R2AI_MERGE_KEEP")   # mặc định xoá đầu vào cũ (v>=3) cho đỡ tốn ổ; R2AI_MERGE_KEEP=1 để giữ
    # 1) doc mới vào data/docs (fuse_dense / ltr đọc văn bản & tiêu đề từ đây)
    dst = f"data/docs/corpus-{a.tag}.parquet"
    shutil.copy(f"{a.docs}/new_docs.parquet", dst); print("docs ->", dst)
    # 2) embedding GN doc-level cho PRF (ltr.py đọc mọi data/cache/dense_gn/part*.npy)
    np.save(f"data/cache/dense_gn/part{a.tag}.npy", np.load(f"{K}/gn_docs.npy"))
    shutil.copy(f"{K}/gn_ids.json", f"data/cache/dense_gn/part{a.tag}.ids.json")
    # 3) gộp list dense: top-300 theo điểm trên (cũ ∪ mới)
    for k in ("d1", "d2", "d3"):
        old, new = json.load(open(OLD[k])), json.load(open(f"{K}/{k}_new.json"))
        out = {}
        for q in old:
            best = {}
            for x in old[q] + new.get(q, []):
                if str(x[1]) not in best or x[0] > best[str(x[1])][0]:
                    best[str(x[1])] = x
            out[q] = sorted(best.values(), key=lambda x: -x[0])[:300]
        nn = sum(str(x[1]) in {str(y[1]) for y in new.get(q, [])} for q in out for x in out[q])
        json.dump(out, open(NEW[k], "w"), ensure_ascii=False); print(k, "->", NEW[k], f"new-doc entries {nn} ({nn / len(out):.1f}/câu)")
        if RM:   # ổ gần đầy: bỏ đầu vào ngay khi đã ghi xong bản mới
            os.remove(f"{K}/{k}_new.json"); os.remove(OLD[k])
    # 4) e5: nối embedding doc mới vào bản cũ
    os.makedirs(NEW["e5"], exist_ok=True)
    ids = json.load(open(f"{OLD['e5']}/e5_ids.json")) + json.load(open(f"{K}/e5_ids.json"))
    np.save(f"{NEW['e5']}/e5_docs.npy", np.concatenate([np.load(f"{OLD['e5']}/e5_docs.npy"), np.load(f"{K}/e5_docs.npy")]))
    json.dump(ids, open(f"{NEW['e5']}/e5_ids.json", "w"))
    for f in ("e5_queries.npy", "e5_qids.json"):
        shutil.copy(f"{OLD['e5']}/{f}", f"{NEW['e5']}/{f}")
    if RM:
        shutil.rmtree(OLD["e5"])
    # 5) điểm cross-encoder / GN fine-tune cho cặp (câu, doc mới) – model train trên tất cả nhãn (kaggle_savemodels)
    P = pd.read_parquet(f"{K}/pairs.parquet"); P["doc_id"] = P.doc_id.astype(str)
    for col, name in RR.items():
        if col not in P.columns:
            print("thiếu cột", col, "– bỏ qua"); continue
        R = pd.read_parquet(OLD["rr"](name))[["qid", "doc_id", "s"]]; R["doc_id"] = R.doc_id.astype(str)
        pd.concat([R, P[["qid", "doc_id"]].assign(s=P[col])]).drop_duplicates(["qid", "doc_id"]).to_parquet(NEW["rr"](name), index=False)
    print("pairs", len(P))
    v3 = NEW["rr"]
    print(f"ltr.py --preds data/cache/vi_preds_300.jsonl --scores data/cache/rr_gn_top200.npy --dense {NEW['d1']} --dense2 {NEW['d2']} "
          f"--dense3 {NEW['d3']} --e5 {NEW['e5']} --rr {v3('rr_bge_all')} --rr2 {v3('rr_win')} --rr3 {v3('ce_ft')} --rr4 {v3('ce_ftwin')} "
          f"--gnft {v3('gnft')} --stages 2 --seeds 5 --out data/cache/ltr_order_v{a.ver}.json")


if __name__ == "__main__":
    main()
