"""Hợp nhất ứng viên: (BM25 → GreenNode chunk rerank) ⊕ (GreenNode dense toàn corpus) bằng RRF, rồi ghi zip nộp.

    python scripts/fuse_dense.py --preds data/cache/vi_preds_300.jsonl --scores data/cache/rr_gn_top200.npy --top 200 \
        --dense data/cache/dense_gn_top300.json --w-dense 1.0 --n-docs 120 --n-chunks 80 --out submissions/v9.zip
    (--eval-only: chỉ đếm doc đúng-đã-biết trong top-n từ data/eval/details_*.jsonl, không ghi zip)
"""
import argparse
import glob
import json
import sys
import zipfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "scripts")
from chunk_v2 import select_chunks  # noqa: E402
from rerank_vi import load_preds, pairs  # noqa: E402


def fused_lists(a):
    P = pairs(load_preds(a.preds), a.top)
    S = np.load(a.scores)
    df = pd.DataFrame(P, columns=["qid", "q", "r", "doc", "text"]).assign(s=S)
    dense = json.load(open(a.dense))
    dense2 = json.load(open(a.dense2)) if a.dense2 else {}
    dense3 = json.load(open(a.dense3)) if a.dense3 else {}   # dense cấp đoạn: [score, doc_id, chunk_text]
    plans = {int(json.loads(l)["id"]): json.loads(l) for l in open("data/cache/query_plans.jsonl")}
    out = {}
    for qid, g in df.groupby("qid"):
        n = len(g)
        f = a.alpha * g.s.rank(ascending=False) / n + (1 - a.alpha) * (g.r + 1) / n
        g = g.assign(f=f).sort_values("f")
        rr = {}
        for i, d in enumerate(g.doc):
            rr[d] = rr.get(d, 0) + 1 / (a.k + i + 1)
        for i, (_, d) in enumerate(dense.get(str(qid), [])[:a.dense_k]):
            rr[str(d)] = rr.get(str(d), 0) + a.w_dense / (a.k + i + 1)
        for i, x in enumerate(dense2.get(str(qid), [])[:a.dense_k]):   # [score, doc] hoặc [score, doc, text]
            rr[str(x[1])] = rr.get(str(x[1]), 0) + a.w_dense2 / (a.k + i + 1)
        ch3 = {}
        for i, x in enumerate(dense3.get(str(qid), [])[:a.dense_k]):   # [score, doc, chunk_text]; vài mục chỉ có [score, doc]
            d = x[1]; rr[str(d)] = rr.get(str(d), 0) + a.w_dense3 / (a.k + i + 1)
            if len(x) > 2 and x[2]:
                ch3[str(d)] = x[2]
        docs = sorted(rr, key=lambda d: -rr[d])
        chm = dict(ch3); chm.update(dict(zip(g.doc, g.text)))     # ưu tiên chunk đã chấm GreenNode từ BM25, sau đó chunk dense-đoạn
        out[int(qid)] = (docs, chm, g.q.iloc[0], plans.get(int(qid)))
    return out


def known_pos():
    pos = {}
    for f in glob.glob("data/eval/details_*.jsonl"):
        for l in open(f):
            d = json.loads(l)
            for p in d["chunk"]["predictions"]:
                if p["relevant"]:
                    pos.setdefault(int(d["id"]), set()).add(p["doc_id"].split(":")[1])
    return pos


def expand(doc, c, n):
    """Nới chunk c (nguyên văn trong doc) thành đoạn liền mạch dài ~n ký tự: ưu tiên mở về phía sau, phần thiếu mở về trước."""
    p = doc.find(c)
    if p < 0:
        return c
    end = min(len(doc), p + n)
    st = max(0, end - n)
    if st > 0:   # cắt ở ranh giới khoảng trắng để không cắt đôi âm tiết
        sp = doc.rfind(" ", 0, st); st = sp + 1 if sp >= 0 and st - sp < 50 else st
    if end < len(doc):
        sp = doc.find(" ", end); end = sp if 0 <= sp < end + 50 else end
    return doc[st:end]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True); ap.add_argument("--scores", required=True)
    ap.add_argument("--top", type=int, default=200); ap.add_argument("--alpha", type=float, default=0.85)
    ap.add_argument("--dense", required=True); ap.add_argument("--dense-k", type=int, default=300)
    ap.add_argument("--w-dense", type=float, default=1.0)
    ap.add_argument("--dense2", default=""); ap.add_argument("--w-dense2", type=float, default=1.0)
    ap.add_argument("--dense3", default=""); ap.add_argument("--w-dense3", type=float, default=1.0); ap.add_argument("--k", type=int, default=60)
    ap.add_argument("--n-docs", type=int, default=120); ap.add_argument("--n-chunks", type=int, default=80)
    ap.add_argument("--out"); ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--adaptive", type=float, default=0, help=">0: số doc/câu = hệ số × số gold dự đoán (40..200)")
    ap.add_argument("--chunk-ratio", type=float, default=0.67)
    ap.add_argument("--chunk-max", type=int, default=130)
    ap.add_argument("--extra-chunks", type=int, default=0, help="K doc đầu mỗi câu thêm 1 đoạn thứ hai (khác đoạn đầu)")
    ap.add_argument("--chunk-chars", type=int, default=0, help=">0: nới mỗi chunk (nguyên văn, liền mạch) tới N ký tự")
    ap.add_argument("--order", default="", help="json {qid: [doc...]} từ scripts/ltr.py thay thứ tự RRF")
    ap.add_argument("--order-chunks-only", action="store_true", help="--order chỉ dùng để chọn chunk; danh sách doc giữ RRF")
    ap.add_argument("--ef2", type=float, default=0, help=">0: số doc/câu = argmax F2 kỳ vọng từ xác suất LTR (<order>_probs.json), G = hệ số × Σp")
    ap.add_argument("--ef2-range", type=int, nargs=2, default=[30, 250])
    ap.add_argument("--dump", default="", help="ghi <dump>_rrf.json (thứ tự doc RRF top300) + <dump>_ndocs.json")
    ap.add_argument("--lzma", action="store_true", help="nén ZIP_LZMA (nhỏ hơn ~40%%, BTC đọc được – v27)")
    a = ap.parse_args()
    L = fused_lists(a)
    if a.order:
        od = json.load(open(a.order))
        L = {q: (od.get(str(q), v[0]), *v[1:], v[0]) for q, v in L.items()}
    if a.eval_only:
        pos = known_pos()
        for n in (60, 120):
            print(f"w_dense={a.w_dense} known-pos in top{n}:",
                  round(sum(len(set(L[q][0][:n]) & pos.get(q, set())) for q in L) / len(pos), 3))
        return
    need = {d for q, (docs, ch, *_ ) in L.items() for i, d in enumerate(docs[:max(130, a.chunk_max)])
            if d not in ch or i < a.extra_chunks or a.chunk_chars}
    texts = {}
    for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "text"]).to_pandas()
        t = t[t.doc_id.astype(str).isin(need)]
        texts.update(zip(t.doc_id.astype(str), t.text))
    nd, nc = {}, {}
    if a.adaptive:
        # số doc theo từng câu: điểm dense hạng 50 → phân vị → số gold dự đoán (map từ public details) × hệ số
        mp = json.load(open("data/cache/gold_count_map.json")); xs, ys = np.array(mp["xs"]), np.array(mp["ys"])
        dn = json.load(open(a.dense))
        for qid in L:
            v = dn.get(str(qid), [])
            s50 = v[49][0] if len(v) > 49 else (v[-1][0] if v else 0)
            pred = np.quantile(ys, min(1.0, np.searchsorted(xs, s50) / len(xs)))
            nd[qid] = int(np.clip(round(a.adaptive * pred), 40, 200))
            nc[qid] = int(np.clip(round(nd[qid] * a.chunk_ratio), 30, a.chunk_max))
    if a.dump:
        json.dump({str(q): (v[4] if len(v) > 4 else v[0])[:300] for q, v in L.items()}, open(a.dump + "_rrf.json", "w"))
        json.dump({str(q): nd.get(q, a.n_docs) for q in L}, open(a.dump + "_ndocs.json", "w"))
        if not a.out:
            return
    if a.ef2:
        pr = json.load(open(a.order.replace(".json", "_probs.json")))
        for qid in L:
            p = np.array(pr.get(str(qid), [])) if str(qid) in pr else None
            if p is None or not len(p):
                continue
            n = np.arange(1, len(p) + 1)
            nd[qid] = int(np.clip(np.argmax(5 * np.cumsum(p) / (4 * a.ef2 * p.sum() + n)) + 1, *a.ef2_range))
            nc[qid] = int(np.clip(round(nd[qid] * a.chunk_ratio), 30, a.chunk_max))
    res = []
    for qid, (docs, ch, qtext, plan, *rrf) in sorted(L.items()):
        doc_list = rrf[0] if (rrf and a.order_chunks_only) else docs
        n_docs, n_chunks = nd.get(qid, a.n_docs), nc.get(qid, a.n_chunks)
        chunks = []
        for i, d in enumerate(docs[:n_chunks]):
            c = ch.get(d)
            sel = None
            if not c and d in texts:
                sel = select_chunks(qtext, plan, texts[d], "vi", {"target_chars": 1600})
                c = sel[0] if sel else ""
            if c and a.chunk_chars and len(c) < a.chunk_chars and d in texts:
                c = expand(texts[d], c, a.chunk_chars)
            if c:
                chunks.append({"doc_id": int(d), "chunk_text": c})
            if i < a.extra_chunks and d in texts:   # đoạn thứ hai: ứng viên tốt nhất không trùng đoạn đầu
                sel = select_chunks(qtext, plan, texts[d], "vi", {"target_chars": 1600, "max_chunks": 3})
                c2 = next((x for x in sel if x != c and x[:200] not in (c or "") and (c or "")[:200] not in x), None)
                if c2:
                    chunks.append({"doc_id": int(d), "chunk_text": c2})
        res.append({"id": qid, "relevant_docs": [int(d) for d in doc_list[:n_docs]], "relevant_chunks": chunks})
    ids = set(pd.read_parquet("data/ViBioMIR/query.parquet").id.astype(int))
    res += [{"id": i, "relevant_docs": [], "relevant_chunks": []} for i in sorted(ids - {r["id"] for r in res})]
    res.sort(key=lambda e: e["id"])
    if a.lzma:
        import zip_lzma  # noqa: F401  (vá zipfile: LZMA preset 9e, từ điển 512MB)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_LZMA) if a.lzma else zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        z.writestr("submission.json", json.dumps(res, ensure_ascii=False))
    print("wrote", a.out, len(res))


if __name__ == "__main__":
    main()
