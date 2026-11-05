"""diag13: đo và dựng bài chẩn đoán cho R2AI Stage 3 (chạy CPU trên b660).

1. job1     đo số doc của submission_split.json nằm ngoài top-100 bge (cache_pool_rr.json),
            phân theo hạng trong danh sách nộp + top-100 fhyde so với top-100 bge
2. job2     đo kích thước/thời gian zip ZIP_DEFLATED 9 / ZIP_BZIP2 / ZIP_LZMA + kiểm đọc lại
3. docceil  trần recall doc: relevant_docs = toàn bộ pool (thứ tự fhyde trước, nối pool sau),
            relevant_chunks y nguyên split
4. chunkceil trần recall chunk: qid chẵn = toàn bộ chunk ứng viên (bỏ trùng nguyên văn), qid lẻ = []
5. nqf      n_q theo điểm fused fhyde (ratio fused/fused_top1 hoặc z-score), kẹp [40, 400]

Dùng:
  python scripts/diag13.py job1 --split S.json --pool P.json --fhyde F.json [--out report.json]
  python scripts/diag13.py job2 --split S.json --out-dir DIR
  python scripts/diag13.py docceil --split S.json --pool P.json --fhyde F.json --out submission_docceil.json
  python scripts/diag13.py chunkceil --split S.json --rerank chunk_dw_u900.json.zst --out submission_chunkceil.json
  python scripts/diag13.py nqf --split S.json --pool P.json --fhyde F.json --out submission_nqf.json
"""
import argparse
import io
import json
import statistics
import sys
import time
import zipfile
from pathlib import Path

from chunks_from_rerank import load_rerank
from union_sub import adaptive_n

NQ_LO = 40
NQ_HI = 400
BGE_THR = -2.0
ZIP_LIMIT = 95_000_000
BANDS = ((1, 30, "1-30"), (31, 60, "31-60"), (61, 100, "61-100"), (101, 10 ** 9, "101+"))


def pct(vals: list, p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]


def dist(vals: list) -> dict:
    return {"TB": sum(vals) / len(vals) if vals else float("nan"),
            "trung_vi": pct(vals, 0.5), "p90": pct(vals, 0.9),
            "min": min(vals) if vals else float("nan"), "max": max(vals) if vals else float("nan")}


def rankdata(vals: list) -> list:
    """Hạng trung bình khi có đồng điểm (1-based)."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def spearman(a: list, b: list) -> float:
    """Hệ số tương quan hạng Spearman; 0.0 nếu một trong hai chuỗi không đổi."""
    if len(a) != len(b) or not a:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    da = [x - ma for x in ra]
    db = [x - mb for x in rb]
    na = sum(x * x for x in da) ** 0.5
    nb = sum(x * x for x in db) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return sum(x * y for x, y in zip(da, db)) / (na * nb)


def pick_nq_fused(hits: list, thr: float, lo: int = NQ_LO, hi: int = NQ_HI, method: str = "ratio") -> int:
    """Số doc có điểm fused vượt thr, kẹp [lo, hi].

    method 'ratio': fused / fused_max; 'zscore': (fused - mean) / pstdev trên từng câu.
    """
    vals = [float(h[2]) for h in hits if h[2] is not None]
    if not vals:
        return lo
    if method == "ratio":
        top = max(vals)
        scores = [v / top for v in vals] if top > 0 else [0.0] * len(vals)
    elif method == "zscore":
        sd = statistics.pstdev(vals)
        scores = [0.0] * len(vals) if sd == 0 else [(v - sum(vals) / len(vals)) / sd for v in vals]
    else:
        raise ValueError(f"method la: {method}")
    k = sum(1 for s in scores if s > thr)
    return min(max(k, lo), hi)


def calibrate_fused_thr(per_q_hits: list, method: str, target: float, lo: int = NQ_LO, hi: int = NQ_HI,
                        iters: int = 40) -> tuple[float, list]:
    """Ngưỡng thr để TB n_q ≈ target (nhị phân; mean n_q không tăng theo thr). Trả (thr, danh sách n_q)."""
    lo_t, hi_t = (0.0, 1.0) if method == "ratio" else (-3.0, 6.0)
    best = (float("inf"), lo_t)
    for _ in range(iters):
        mid = (lo_t + hi_t) / 2
        ns = [pick_nq_fused(h, mid, lo, hi, method) for h in per_q_hits]
        m = sum(ns) / max(len(ns), 1)
        if abs(m - target) < best[0]:
            best = (abs(m - target), mid)
        if m > target:
            lo_t = mid
        else:
            hi_t = mid
    thr = round(best[1], 6)
    return thr, [pick_nq_fused(h, thr, lo, hi, method) for h in per_q_hits]


def with_chunk_docs(docs: list, chunks: list) -> tuple[list, int]:
    """docs + doc_id của chunks còn thiếu (theo thứ tự xuất hiện). Trả (docs mới, số doc thêm)."""
    out = list(docs)
    seen = {str(d) for d in out}
    add = 0
    for ch in chunks:
        d = ch["doc_id"]
        if str(d) not in seen:
            seen.add(str(d))
            out.append(d)
            add += 1
    return out, add


def order_pool_docs(pool_hits: list, fhyde_hits: list) -> list:
    """Toàn bộ doc của pool: doc có trong fhyde theo thứ tự fhyde, còn lại nối theo thứ tự pool."""
    pool_ids = [h[0] for h in pool_hits]
    pool_set = {str(d) for d in pool_ids}
    seen, out = set(), []
    for h in fhyde_hits:
        d = h[0]
        if str(d) in pool_set and str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    for d in pool_ids:
        if str(d) not in seen:
            seen.add(str(d))
            out.append(d)
    return out


def dedup_chunk_rows(rows: list, k: int | None = None) -> list:
    """Rows [doc_id, doc_rank, chunk_idx, logit, chunk_text] -> [{doc_id, chunk_text}].

    Sắp logit giảm dần, bỏ trùng nguyên văn (giữ bản đầu), giữ top-k nếu k khác None.
    """
    ordered = sorted(rows, key=lambda r: -float(r[3]))
    seen, out = set(), []
    for r in ordered:
        t = r[4]
        if t in seen:
            continue
        seen.add(t)
        out.append({"doc_id": r[0], "chunk_text": t})
        if k is not None and len(out) >= k:
            break
    return out


def build_docceil(sub: list, pool: dict, fhyde: dict) -> tuple[list, list]:
    """relevant_docs = toàn bộ pool (fhyde trước, pool sau) + doc chunk còn thiếu; chunk y nguyên split."""
    po = {str(k): v for k, v in pool.items()}
    fu = {str(k): v for k, v in fhyde.items()}
    out, added = [], []
    for it in sub:
        q = str(it["id"])
        docs = order_pool_docs(po.get(q, []), fu.get(q, []))
        docs, add = with_chunk_docs(docs, it["relevant_chunks"])
        added.append(add)
        out.append({"id": it["id"], "relevant_docs": docs, "relevant_chunks": it["relevant_chunks"]})
    return out, added


def build_chunkceil(sub: list, rerank: dict, k: int | None = None) -> tuple[list, dict]:
    """qid chẵn: toàn bộ chunk ứng viên (bỏ trùng nguyên văn, giữ top-k theo logit); qid lẻ: [].

    relevant_docs = docs của split + doc nguồn chunk còn thiếu. Trả (out, stats).
    """
    rr = {str(q): v for q, v in rerank.items()}
    out, added = [], []
    nc_even, n_even, n_odd, removed = [], 0, 0, 0
    for it in sub:
        qid = it["id"]
        if qid % 2 == 0:
            rows = rr.get(str(qid), [])
            chunks = dedup_chunk_rows(rows, k)
            removed += max(len(rows) - len(chunks), 0)
            n_even += 1
            nc_even.append(len(chunks))
        else:
            chunks = []
            n_odd += 1
        docs, add = with_chunk_docs(it["relevant_docs"], chunks)
        added.append(add)
        out.append({"id": qid, "relevant_docs": docs, "relevant_chunks": chunks})
    stats = {"cau_chan": n_even, "cau_le": n_odd,
             "TB_chunk_nua_chan": sum(nc_even) / len(nc_even) if nc_even else 0.0,
             "min_chunk_nua_chan": min(nc_even) if nc_even else 0,
             "max_chunk_nua_chan": max(nc_even) if nc_even else 0,
             "chunk_trung_bo": removed, "TB_doc_them": sum(added) / max(len(added), 1),
             "tong_doc_them": sum(added)}
    return out, stats


def build_nqf(sub: list, pool: dict, fhyde: dict, thr: float, method: str = "ratio",
              lo: int = NQ_LO, hi: int = NQ_HI) -> tuple[list, list, list]:
    """n_q theo điểm fused; doc nền = n_q doc đầu theo thứ tự fhyde + doc chunk còn thiếu; chunk y nguyên split."""
    po = {str(k): v for k, v in pool.items()}
    fu = {str(k): v for k, v in fhyde.items()}
    out, ns, added = [], [], []
    for it in sub:
        q = str(it["id"])
        hits = fu.get(q, [])
        n = pick_nq_fused(hits, thr, lo, hi, method)
        ns.append(n)
        docs, add = with_chunk_docs([h[0] for h in hits[:n]], it["relevant_chunks"])
        added.append(add)
        out.append({"id": it["id"], "relevant_docs": docs, "relevant_chunks": it["relevant_chunks"]})
    return out, ns, added


def job1_stats(sub: list, pool: dict, fhyde: dict) -> dict:
    """Số doc ngoài top-100 bge mỗi câu, phân theo hạng nộp; top-100 fhyde so với top-100 bge."""
    po = {str(k): v for k, v in pool.items()}
    fu = {str(k): v for k, v in fhyde.items()}
    per_out, fhyde_out, docs_lens, nq_old, pool_lens, extra_lens = [], [], [], [], [], []
    band_out = {k: 0 for _, _, k in BANDS}
    band_chunk = {k: 0 for _, _, k in BANDS}
    band_tot = {k: 0 for _, _, k in BANDS}
    for it in sub:
        q = str(it["id"])
        docs = it["relevant_docs"]
        chunk_docs = {str(c["doc_id"]) for c in it["relevant_chunks"]}
        hits = po.get(q, [])
        top100 = {str(h[0]) for h in hits[:100]}
        out_cnt = 0
        for i, d in enumerate(docs):
            r = i + 1
            key = BANDS[-1][2]
            for a, b, kk in BANDS:
                if a <= r <= b:
                    key = kk
                    break
            band_tot[key] += 1
            if str(d) not in top100:
                out_cnt += 1
                band_out[key] += 1
            if str(d) in chunk_docs:
                band_chunk[key] += 1
        per_out.append(out_cnt)
        docs_lens.append(len(docs))
        fhyde_hits = fu.get(q, [])
        fhyde_out.append(sum(1 for h in fhyde_hits[:100] if str(h[0]) not in top100))
        pool_lens.append(len(hits))
        extra_lens.append(sum(1 for h in fhyde_hits if str(h[0]) not in {str(x[0]) for x in hits}))
        nq_old.append(adaptive_n(hits, BGE_THR, NQ_LO, NQ_HI))
    nq = max(len(sub), 1)
    theo_hang = {}
    for _, _, key in BANDS:
        theo_hang[key] = {"TB_doc": band_tot[key] / nq,
                          "TB_ngoai_top100": band_out[key] / nq,
                          "ti_le_ngoai": band_out[key] / band_tot[key] if band_tot[key] else 0.0,
                          "TB_doc_co_chunk": band_chunk[key] / nq}
    return {"cau": len(sub),
            "doc_phep": dist(docs_lens),
            "ngoai_top100_bge": dist(per_out),
            "theo_hang": theo_hang,
            "fhyde_top100_ngoai_top100_bge": dist(fhyde_out),
            "n_q_cu_logit_bge": dist(nq_old),
            "pool_len": dist(pool_lens),
            "fhyde_ngoai_pool": dist(extra_lens)}


def job2_zips(split_path: str, out_dir: str) -> dict:
    """Nén submission_split.json bằng DEFLATED 9 / BZIP2 / LZMA; kiểm đọc lại zipfile + json.load."""
    src = Path(split_path)
    out = Path(out_dir)
    raw = src.read_bytes()
    parsed = json.loads(raw)
    kinds = [("ZIP_DEFLATED", "submission_split_defl9.zip", zipfile.ZIP_DEFLATED, 9),
             ("ZIP_BZIP2", "submission_split_bz2.zip", zipfile.ZIP_BZIP2, None),
             ("ZIP_LZMA", "submission_split_lzma.zip", zipfile.ZIP_LZMA, None)]
    rep = {"json_bytes": len(raw), "zip": []}
    for method, name, ztype, lvl in kinds:
        p = out / name
        t0 = time.time()
        with zipfile.ZipFile(p, "w", ztype, compresslevel=lvl) as z:
            z.write(src, arcname="submission.json")
        sec = time.time() - t0
        with zipfile.ZipFile(p) as z:
            data = z.read("submission.json")
            obj = json.load(io.BytesIO(data))
        rec = {"method": method, "file": name, "bytes": p.stat().st_size, "sec": round(sec, 1),
               "doc_lai": "OK", "bang_goc_bytes": data == raw, "bang_goc_json": obj == parsed}
        del data, obj
        rep["zip"].append(rec)
    return rep


def cmd_job1(a) -> None:
    sub = json.loads(Path(a.split).read_text(encoding="utf-8"))
    pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))
    fhyde = json.loads(Path(a.fhyde).read_text(encoding="utf-8"))
    rep = job1_stats(sub, pool, fhyde)
    txt = json.dumps(rep, ensure_ascii=False)
    print(txt, flush=True)
    if a.out:
        Path(a.out).write_text(txt, encoding="utf-8")


def cmd_job2(a) -> None:
    rep = job2_zips(a.split, a.out_dir)
    txt = json.dumps(rep, ensure_ascii=False)
    print(txt, flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")


def cmd_docceil(a) -> None:
    sub = json.loads(Path(a.split).read_text(encoding="utf-8"))
    pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))
    fhyde = json.loads(Path(a.fhyde).read_text(encoding="utf-8"))
    out, added = build_docceil(sub, pool, fhyde)
    del sub, pool, fhyde
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    nd = [len(s["relevant_docs"]) for s in out]
    rep = {"doc_moi_cau": dist(nd), "doc_them_cho_chunk": dist(added), "tong_doc_them": sum(added)}
    txt = json.dumps(rep, ensure_ascii=False)
    print(f"ghi {a.out}: {len(out)} cau, doc/cau TB {sum(nd) / max(len(nd), 1):.2f} "
          f"min {min(nd)} max {max(nd)}, {txt}", flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")


def cmd_chunkceil(a) -> None:
    sub = json.loads(Path(a.split).read_text(encoding="utf-8"))
    rerank = load_rerank(a.rerank)
    out, st = build_chunkceil(sub, rerank)
    k_full = max((len(it["relevant_chunks"]) for it in out if it["id"] % 2 == 0), default=0)

    def write_probe(out_list: list) -> int:
        Path(a.out).write_text(json.dumps(out_list, ensure_ascii=False), encoding="utf-8")
        with zipfile.ZipFile(a.probe_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
            z.write(a.out, arcname="submission.json")
        return Path(a.probe_zip).stat().st_size

    k = k_full
    size = write_probe(out)
    if size > a.zip_limit and k_full > 0:
        lo, hi, best_k = 1, k_full, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            out_k, _ = build_chunkceil(sub, rerank, k=mid)
            if write_probe(out_k) <= a.zip_limit:
                best_k, lo = mid, mid + 1
            else:
                hi = mid - 1
        k = best_k
        out, st = build_chunkceil(sub, rerank, k=k)
        size = write_probe(out)
    Path(a.probe_zip).unlink()
    st["k_giu"] = k
    st["zip_probe_bytes"] = size
    st["duoi_95MB"] = size <= a.zip_limit
    txt = json.dumps(st, ensure_ascii=False)
    print(f"ghi {a.out}: {len(out)} cau, {txt}", flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")


def cmd_nqf(a) -> None:
    sub = json.loads(Path(a.split).read_text(encoding="utf-8"))
    pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))
    fhyde = json.loads(Path(a.fhyde).read_text(encoding="utf-8"))
    po = {str(k): v for k, v in pool.items()}
    fu = {str(k): v for k, v in fhyde.items()}
    nq_old = [adaptive_n(po.get(str(it["id"]), []), BGE_THR, NQ_LO, NQ_HI) for it in sub]
    per_q = [fu.get(str(it["id"]), []) for it in sub]
    target = a.target if a.target is not None else sum(nq_old) / max(len(nq_old), 1)
    cands = {}
    for method in ("ratio", "zscore"):
        thr_m, ns_m = calibrate_fused_thr(per_q, method, target)
        cands[method] = {"thr": thr_m, "n_q": ns_m, "TB_n_q": sum(ns_m) / max(len(ns_m), 1),
                         "spearman_vs_cu": spearman(ns_m, nq_old)}
    if a.method:
        method = a.method
    else:
        method = max(cands, key=lambda m: (cands[m]["spearman_vs_cu"], m == "ratio"))
    thr = a.thr if a.thr is not None else cands[method]["thr"]
    out, ns_new, added = build_nqf(sub, pool, fhyde, thr, method)
    split_docs = {str(it["id"]): it["relevant_docs"] for it in sub}
    khac, giao = [], []
    for s in out:
        a_, b_ = {str(d) for d in s["relevant_docs"]}, {str(d) for d in split_docs[str(s["id"])]}
        khac.append(len(a_ ^ b_))
        giao.append(len(a_ & b_))
    nd = [len(s["relevant_docs"]) for s in out]
    rep = {"phuong_phap": method, "nguong": thr, "target_TB_n_q": target,
           "n_q_moi": dist(ns_new), "n_q_cu": dist(nq_old),
           "spearman_n_q": spearman(ns_new, nq_old),
           "doc_moi_cau": dist(nd), "doc_split_moi_cau": dist([len(v) for v in split_docs.values()]),
           "TB_doc_khac_nhau": sum(khac) / max(len(khac), 1), "TB_giao": sum(giao) / max(len(giao), 1),
           "TB_doc_them_cho_chunk": sum(added) / max(len(added), 1),
           "cach_do_cac_method": {m: {kk: vv for kk, vv in cands[m].items() if kk != "n_q"} for m in cands}}
    del sub, pool, fhyde
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    txt = json.dumps(rep, ensure_ascii=False)
    print(f"ghi {a.out}: {len(out)} cau, {txt}", flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    j1 = sp.add_parser("job1")
    j1.add_argument("--split", required=True)
    j1.add_argument("--pool", required=True)
    j1.add_argument("--fhyde", required=True)
    j1.add_argument("--out", default="")
    j2 = sp.add_parser("job2")
    j2.add_argument("--split", required=True)
    j2.add_argument("--out-dir", required=True)
    j2.add_argument("--report", default="")
    dc = sp.add_parser("docceil")
    dc.add_argument("--split", required=True)
    dc.add_argument("--pool", required=True)
    dc.add_argument("--fhyde", required=True)
    dc.add_argument("--out", required=True)
    dc.add_argument("--report", default="")
    cc = sp.add_parser("chunkceil")
    cc.add_argument("--split", required=True)
    cc.add_argument("--rerank", required=True)
    cc.add_argument("--out", required=True)
    cc.add_argument("--report", default="")
    cc.add_argument("--zip-limit", type=int, default=ZIP_LIMIT)
    cc.add_argument("--probe-zip", default="")
    nq = sp.add_parser("nqf")
    nq.add_argument("--split", required=True)
    nq.add_argument("--pool", required=True)
    nq.add_argument("--fhyde", required=True)
    nq.add_argument("--out", required=True)
    nq.add_argument("--report", default="")
    nq.add_argument("--method", choices=["ratio", "zscore"], default=None)
    nq.add_argument("--thr", type=float, default=None)
    nq.add_argument("--target", type=float, default=None)
    a = ap.parse_args()
    if a.cmd == "chunkceil" and not a.probe_zip:
        a.probe_zip = str(Path(a.out).with_suffix("")) + ".probe.zip"
    {"job1": cmd_job1, "job2": cmd_job2, "docceil": cmd_docceil,
     "chunkceil": cmd_chunkceil, "nqf": cmd_nqf}[a.cmd](a)


if __name__ == "__main__":
    main()
