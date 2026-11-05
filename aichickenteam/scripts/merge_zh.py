"""Trộn doc ZH (xác suất từ scripts/zh_ltr.py) vào bài VI đã có (zip, vd v65): mỗi câu gộp (doc VI, p LTR) ∪ (doc ZH, p ZH), sắp theo p,
số doc = argmax F2 kỳ vọng (như fuse_dense --ef2), chunk theo cùng thứ tự (VI: chunk sẵn trong zip gốc; ZH: nguyên văn text[:--zh-chars]), tối đa --chunk-max.
    python scripts/merge_zh.py --base submissions/v65.zip --order data/cache/ltrx7_out/ltr_order_oof34 --zh data/cache/zh_probs.parquet --out data/cache/tmp/v70_deflate.zip [--sim]
--sim: chỉ in F2 docs mô phỏng (nhãn hiện có, doc chưa thấy = sai) cho bài gốc vs bài trộn.
"""
import argparse, json, sys, zipfile
import numpy as np, pandas as pd
sys.path.insert(0, "scripts")
from ltr import labels  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--order", required=True); ap.add_argument("--zh", required=True); ap.add_argument("--out", default="")
ap.add_argument("--ef2", type=float, default=1.9); ap.add_argument("--lo", type=int, default=30); ap.add_argument("--hi", type=int, default=300)
ap.add_argument("--chunk-max", type=int, default=240); ap.add_argument("--zh-chars", type=int, default=3000); ap.add_argument("--zh-top", type=int, default=60)
ap.add_argument("--zh-scale", type=float, default=1.0, help="nhân xác suất ZH (hiệu chỉnh)"); ap.add_argument("--sim", action="store_true")
ap.add_argument("--vi-fill", type=int, default=0, help=">0: doc VI được chọn mà zip gốc KHÔNG có chunk → chunk = cửa sổ d3 (hoặc đầu doc) nới tới N ký tự")
ap.add_argument("--d3", default="data/cache/dense_chunks_top_v9.json")
ap.add_argument("--vi-mode", default="trunc", choices=["trunc", "center", "docstart"], help="trunc: N ký tự đầu chunk gốc (bắt đầu ở cửa sổ dense); center: căn giữa cửa sổ dense (cần --vi-texts); docstart: N ký tự đầu doc")
ap.add_argument("--vi-texts", default="", help="thư mục corpus-*.parquet (doc_id, text) cho --vi-mode center/docstart")
ap.add_argument("--vi-dual", type=int, default=0, help=">0: thêm chunk NGẮN N ký tự căn giữa cửa sổ dense cho doc VI (bắt gold ngắn – luật chấm mới 23/10)")
ap.add_argument("--dual-top", type=int, default=150)
ap.add_argument("--vi-chars", type=int, default=0, help=">0: cắt chunk VI (nới về sau từ cửa sổ dense, 8000 ký tự) còn N ký tự đầu ⇒ tiết kiệm dung lượng để tăng --chunk-max")
a = ap.parse_args()
z = zipfile.ZipFile(a.base); B = {r["id"]: r for r in json.loads(z.read(z.namelist()[0]))}
od = json.load(open(a.order + ".json")); pr = json.load(open(a.order + "_probs.json"))
Z = pd.read_parquet(a.zh); Z["doc_id"] = Z.doc_id.astype(str)
ZG = {q: g.sort_values("p", ascending=False).head(a.zh_top) for q, g in Z.groupby("qid")}
need = set(Z.doc_id)
TX = {}
if not a.sim:
    D = pd.read_parquet("data/cache/kg_zhdocs/zh_docs.parquet", columns=["doc_id", "text"]); D["doc_id"] = D.doc_id.astype(str)
    TX = dict(zip(D.doc_id, D.text.fillna("")))
try:   # nhãn chỉ để IN F2 mô phỏng; thiếu file details (người tái hiện) vẫn chạy được
    pos, seen = labels()
except Exception:
    pos, seen = {}, {}
VIW, VTX = {}, {}
if a.vi_fill and not a.sim:   # văn bản VI cho doc thiếu chunk
    import glob, pyarrow.parquet as pq
    sys.path.insert(0, "scripts"); from fuse_dense import expand
    need_vi = set()
    for q, b in B.items():
        have = {str(c["doc_id"]) for c in b["relevant_chunks"]}
        need_vi |= {d for d in od.get(str(q), [])[:320] if d not in have}
    for f in sorted(glob.glob("data/docs/corpus-*.parquet")):
        t = pq.read_table(f, columns=["doc_id", "text"]).to_pandas(); t["doc_id"] = t.doc_id.astype(str); t = t[t.doc_id.isin(need_vi)]
        VTX.update(zip(t.doc_id, t.text.fillna("")))
    D3 = json.load(open(a.d3)); VIW = {(int(q), str(x[1])): x[2] for q, v in D3.items() for x in v if len(x) > 2 and x[2]}; del D3
    print("VI cần bù chunk:", len(need_vi), "có text:", len(VTX), flush=True)
import os as _os
GF = "data/eval/details_6294_v65.jsonl"
gold = {int(json.loads(l)["id"]): json.loads(l)["document"]["gold"] for l in open(GF)} if _os.path.exists(GF) else {}


def f2(docs, q):
    g = gold.get(q, 0)
    if not g or not docs: return 0.0
    r = sum(d in pos[q] for d in docs); P, R = r / len(docs), r / g
    return 5 * P * R / (4 * P + R) if r else 0.0


VFULL = {}
if a.vi_texts and (a.vi_mode != "trunc" or a.vi_dual) and not a.sim:   # văn bản đầy đủ của doc VI có chunk trong bài gốc
    import glob as _g, pyarrow.parquet as _pq
    needv = {str(c["doc_id"]) for b in B.values() for c in b["relevant_chunks"]}
    for f in sorted(_g.glob(f"{a.vi_texts}/**/corpus-*.parquet", recursive=True)):
        t_ = _pq.read_table(f, columns=["doc_id", "text"]).to_pandas(); t_["doc_id"] = t_.doc_id.astype(str); t_ = t_[t_.doc_id.isin(needv)]
        VFULL.update(zip(t_.doc_id, t_.text.fillna("")))
    print("VI full text:", len(VFULL), "/", len(needv), flush=True)
out, s0, s1, nzh, nfill, ndual = [], [], [], [], [], []
for q in sorted(B):
    b = B[q]; vi = [(d, p, "vi") for d, p in zip(od.get(str(q), []), pr.get(str(q), []))]
    zg = ZG.get(q)
    zh = [(d, min(1.0, p * a.zh_scale), "zh") for d, p in zip(zg.doc_id, zg.p)] if zg is not None else []
    allp = sorted(vi + zh, key=lambda x: -x[1])[:400]
    p = np.array([x[1] for x in allp]); n = np.arange(1, len(p) + 1)
    nd = int(np.clip(np.argmax(5 * np.cumsum(p) / (4 * a.ef2 * p.sum() + n)) + 1, a.lo, a.hi)) if len(p) else 0
    sel = allp[:nd]; nzh.append(sum(x[2] == "zh" for x in sel))
    if q in seen:
        s0.append(f2([str(d) for d in b["relevant_docs"]], q)); s1.append(f2([x[0] for x in sel], q))
    if a.sim: continue
    vch = {str(c["doc_id"]): c["chunk_text"] for c in b["relevant_chunks"]}
    ch = []
    for d, _, lang in sel:
        t = vch.get(d) if lang == "vi" else TX.get(d, "")[:a.zh_chars]
        if t and lang == "vi" and a.vi_chars and a.vi_mode != "trunc" and d in VFULL:
            doc = VFULL[d]; N = a.vi_chars
            if a.vi_mode == "docstart":
                st = 0
            else:
                p0 = doc.find(t[:200]); c0 = (p0 if p0 >= 0 else 0) + 800      # cửa sổ dense ~1600 ký tự bắt đầu tại p0
                st = max(0, min(c0 - N // 2, len(doc) - N))
            en = min(len(doc), st + N)
            if st > 0:
                sp = doc.find(" ", st); st = sp + 1 if 0 <= sp < st + 40 else st
            if en < len(doc):
                sp = doc.rfind(" ", st, en); en = sp if sp > en - 40 else en
            t = doc[st:en]
        if t and lang == "vi" and a.vi_chars and len(t) > a.vi_chars:
            t = t[:a.vi_chars]; sp = t.rfind(" "); t = t[:sp] if sp > a.vi_chars - 50 else t   # cắt ở ranh giới từ
        if not t and lang == "vi" and a.vi_fill and d in VTX:
            w = VIW.get((q, d)) or VTX[d][:1600]; t = expand(VTX[d], w, a.vi_fill) if len(w) < a.vi_fill else w[:a.vi_fill]; nfill.append(1)
        if t: ch.append({"doc_id": int(d), "chunk_text": t})
        if lang == "vi" and a.vi_dual and d in VFULL and len(ch) <= a.dual_top and vch.get(d):
            doc = VFULL[d]; p0 = doc.find(vch[d][:200]); c0 = (p0 if p0 >= 0 else 0) + 800; N = a.vi_dual
            if len(doc) > 1.5 * N:
                st = max(0, min(c0 - N // 2, len(doc) - N)); sp = doc.find(" ", st); st = sp + 1 if 0 <= sp < st + 30 else st
                en = min(len(doc), st + N); sp = doc.rfind(" ", st, en); en = sp if sp > en - 30 else en
                ch.append({"doc_id": int(d), "chunk_text": doc[st:en]}); ndual.append(1)
        if len(ch) >= a.chunk_max: break
    out.append({"id": q, "relevant_docs": [int(x[0]) for x in sel], "relevant_chunks": ch})
if s0: print(f"F2 docs mô phỏng (public, nhãn hiện có): gốc {np.mean(s0):.4f} → trộn ZH {np.mean(s1):.4f}  | doc ZH/câu chọn: TB {np.mean(nzh):.1f}")
if not a.sim and a.out:
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zz:
        zz.writestr("submission.json", json.dumps(out, ensure_ascii=False))
    print("chunk VI bù:", len(nfill), "| chunk ngắn thêm:", len(ndual)); print("wrote", a.out, len(out), "| chunk ZH:", sum(1 for r in out for c in r["relevant_chunks"] if str(c["doc_id"]) in need))
