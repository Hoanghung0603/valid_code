"""pool3: ghép doc NEW (ngoài pool) vào pool rerank + dựng cache_fhyde3 + danh sách doc.

Lệnh (CPU, chạy trên b660):
  pairs     doc NEW còn thiếu logit bge (không có trong cache_pool2_rr) -> cands_p3.json
            ({qid: [[doc_id, idx], ...]} cho rerank_run.py); mọi doc NEW -> cands_new_all.json
            (cho dense_pairs.py); 200 cặp đã có logit -> sanity_pairs.json + sanity_ref.json.
  pool3     cache_pool3_rr.json = pool cũ ∪ NEW với logit bge (pool2 trước, thiếu thì --rerank),
            schema giống cache_pool_rr (sort_by_score).
  extend    cache nguồn mở rộng: hit gốc giữ nguyên vị trí, NEW nối đuôi theo điểm giảm dần
            (dùng cho nguồn dense / HyDE của fhyde3); báo số NEW chen lên trên hit gốc.
  doclist   {qid: [doc_id]}: n_q = adaptive_n(logit > T, kẹp [lo, hi]) doc đầu của thứ hạng.
  cmp       tỷ lệ câu khớp thứ tự top-k của hai cache thứ hạng.
  cmpsub    tỷ lệ câu khớp relevant_docs của hai bài nộp.
  cmplogit  so logit hai cache theo (qid, doc_id); exit 1 nếu max |sai| > tol.

Dùng:
  python scripts/pool3.py pairs --pool cache_pool_rr.json --pool2 cache_pool2_rr.json \
      --new new_cands.json --idx-from cache_pool2_rr.json dense_hyde_top200.json ... \
      --out-cands cands_p3.json --out-new-all cands_new_all.json \
      --out-sanity sanity_pairs.json --out-sanity-ref sanity_ref.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from fuse import sort_by_score
from union_sub import adaptive_n

TOP_CMP = 137
N_SANITY_Q = 20
N_SANITY_PER = 5


def load_cache(path) -> dict:
    """Đọc cache {qid: [[doc_id, idx, score], ...]} -> {str(qid): hits}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(q): v for q, v in data.items()}


def hit_map(hits: list) -> dict:
    """{doc_id: hit} theo lần xuất hiện đầu; hit giữ nguyên [doc_id, idx, score]."""
    out = {}
    for h in hits or []:
        out.setdefault(h[0], h)
    return out


def collect_idx(caches: list) -> tuple:
    """({doc_id: idx} từ các cache nguồn, danh sách doc lệch idx giữa các nguồn)."""
    idx_of: dict = {}
    bad: set = set()
    for cache in caches:
        for hits in cache.values():
            for h in hits:
                if h[0] in idx_of and idx_of[h[0]] != h[1]:
                    bad.add(h[0])
                idx_of.setdefault(h[0], h[1])
    return idx_of, sorted(bad)


def missing_pairs(new: dict, known: dict, idx_of: dict) -> tuple:
    """(cặp thiếu logit, mọi cặp NEW, report).

    known = {str(qid): {doc_id: hit}} từ pool ∪ pool2; cặp có logit None trong known
    vẫn tính là thiếu. Bỏ doc không có idx (đếm riêng).
    """
    cands_m, cands_a = {}, {}
    n_m = n_a = n_have = 0
    no_idx: set = set()
    for q, docs in new.items():
        kh = known.get(q, {})
        ml, al = [], []
        for d in docs:
            i = idx_of.get(d)
            if i is None:
                no_idx.add(d)
                continue
            al.append([d, i])
            n_a += 1
            h = kh.get(d)
            if h is not None and h[2] is not None:
                n_have += 1
            else:
                ml.append([d, i])
                n_m += 1
        cands_a[q] = al
        if ml:
            cands_m[q] = ml
    n = max(len(new), 1)
    rep = {"so_cau": len(new), "cap_thieu_logit": n_m, "TB_thieu_cau": n_m / n,
           "cap_new": n_a, "TB_new_cau": n_a / n, "da_co_logit": n_have,
           "so_thieu_idx": len(no_idx), "thieu_idx_mau": sorted(no_idx)[:20]}
    return cands_m, cands_a, rep


def sanity_pairs(pool: dict, p2: dict, n_q: int = N_SANITY_Q, per: int = N_SANITY_PER) -> tuple:
    """(cặp kiểm tra, logit tham chiếu): per doc đầu của pool + per doc NEW của pool2 mỗi câu."""
    pairs, ref = {}, {}
    for q in sorted(pool, key=lambda x: int(x))[:n_q]:
        ph = hit_map(pool[q])
        old = [h for h in pool[q] if h[2] is not None][:per]
        new = [h for h in p2.get(q, []) if h[0] not in ph and h[2] is not None][:per]
        rows = old + new
        if not rows:
            continue
        pairs[q] = [[h[0], h[1]] for h in rows]
        ref[q] = {str(h[0]): h[2] for h in rows}
    return pairs, ref


def build_pool3(pool: dict, new: dict, known: dict, rerank: dict, idx_of: dict) -> tuple:
    """pool cũ (giữ hit) ∪ NEW có logit (known trước, rồi rerank); sort_by_score. Trả (cache, report).

    known = {qid: {doc_id: hit}} (hit-map); rerank = cache {qid: [[doc_id, idx, logit], ...]}.
    """
    out: dict = {}
    errs: list = []
    sizes_old, sizes_add, n_none = [], [], 0
    for q, hits in pool.items():
        merged = [list(h) for h in hits]
        seen = {h[0] for h in hits}
        kh, rh = known.get(q, {}), hit_map(rerank.get(q, []))
        for d in new.get(q, []):
            if d in seen:
                errs.append(f"cau {q}: doc {d} trung voi pool")
                continue
            seen.add(d)
            h = kh.get(d)
            if h is None or h[2] is None:
                h = rh.get(d)
            if h is None or h[2] is None:
                merged.append([d, idx_of.get(d), None])
                n_none += 1
            else:
                merged.append([h[0], h[1], h[2]])
        sizes_old.append(len(hits))
        sizes_add.append(len(merged) - len(hits))
        out[q] = sort_by_score(merged)
    n = max(len(out), 1)
    rep = {"so_cau": len(out), "TB_pool_cu": sum(sizes_old) / n,
           "TB_new_them": sum(sizes_add) / n, "TB_pool3": sum(len(v) for v in out.values()) / n,
           "new_thieu_logit_None": n_none, "loi": errs[:20], "so_loi": len(errs)}
    return out, rep


def extend_source(base: dict, new_scores: dict) -> tuple:
    """hit gốc giữ vị trí + NEW (không trùng) nối đuôi theo điểm giảm dần. Trả (cache, report)."""
    out: dict = {}
    n_add, viol, viol_mau = [], 0, []
    for q in sorted(set(base) | set(new_scores), key=lambda x: int(x)):
        bh = [list(h) for h in base.get(q, [])]
        seen = {h[0] for h in bh}
        add = [list(h) for h in new_scores.get(q, []) if h[0] not in seen]
        add.sort(key=lambda h: (h[2] is None, 0.0 if h[2] is None else -h[2]))
        base_min = bh[-1][2] if bh else None
        if base_min is not None:
            for h in add:
                if h[2] is not None and h[2] > base_min:
                    viol += 1
                    if len(viol_mau) < 10:
                        viol_mau.append([q, h[0], h[2], base_min])
        n_add.append(len(add))
        out[q] = bh + add
    n = max(len(out), 1)
    rep = {"so_cau": len(out), "TB_new_noi_duoi": sum(n_add) / n,
           "new_chen_len_tren_goc": viol, "mau_chen": viol_mau}
    return out, rep


def doclist(pool3: dict, rank: dict, new_ids: dict, meta: dict,
            thr: float, lo: int, hi: int) -> tuple:
    """(danh sách doc theo luật n_q, report): doc = n_q doc đầu của rank; thống kê doc NEW."""
    out: dict = {}
    ns, n_new, src_tb = [], [], {}
    nsrc_tb = 0.0
    for q in sorted(pool3, key=lambda x: int(x)):
        n = adaptive_n(pool3[q], thr, lo, hi)
        docs, seen = [], set()
        for h in rank.get(q, []):
            d = h[0]
            if d in seen:
                continue
            seen.add(d)
            docs.append(d)
            if len(docs) >= n:
                break
        out[str(q)] = docs
        ns.append(n)
        nset = new_ids.get(q, set())
        names = meta.get(q, {})
        hit_new = 0
        for d in docs:
            if d not in nset:
                continue
            hit_new += 1
            for s in names.get(d, []):
                src_tb[s] = src_tb.get(s, 0.0) + 1
        n_new.append(hit_new)
        nsrc_tb += sum(len(names.get(d, [])) for d in docs if d in nset)
    n = max(len(out), 1)
    rep = {"so_cau": len(out), "thr": thr, "lo": lo, "hi": hi,
           "TB_doc_cau": sum(ns) / n, "min_doc": min(ns), "max_doc": max(ns),
           "TB_new_cau": sum(n_new) / n,
           "phan_bo_nguon_new": {s: v / n for s, v in sorted(src_tb.items())},
           "TB_so_nguon_moi_new": nsrc_tb / max(sum(n_new), 1)}
    return out, rep


def cmp_cache(a: dict, b: dict, top: int = TOP_CMP) -> dict:
    """So thứ tự top-k (doc_id) hai cache; trả report (tỷ lệ câu khớp, vị trí lệch đầu)."""
    qs = [q for q in a if q in b]
    match, first_diff = 0, []
    for q in qs:
        la = [h[0] if isinstance(h, (list, tuple)) else h for h in a[q][:top]]
        lb = [h[0] if isinstance(h, (list, tuple)) else h for h in b[q][:top]]
        if la == lb:
            match += 1
        else:
            fd = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), min(len(la), len(lb)))
            first_diff.append(fd)
    n = max(len(qs), 1)
    return {"top": top, "so_cau_khop": match, "so_cau": len(qs),
            "ty_le_khop": match / n,
            "lech_dau_trung_vi": sorted(first_diff)[len(first_diff) // 2] if first_diff else None,
            "cau_thieu": len(set(a) ^ set(b))}


def cmp_sub(a: list, b: list) -> dict:
    """So relevant_docs hai bài nộp; trả report (tỷ lệ khớp tuyệt đối, TB |giao|/|ref|)."""
    bmap = {str(it["id"]): it for it in b}
    exact, ratio, missing = 0, [], 0
    for it in a:
        ref = bmap.get(str(it["id"]))
        if ref is None:
            missing += 1
            continue
        da = [str(d) for d in it["relevant_docs"]]
        db = [str(d) for d in ref["relevant_docs"]]
        if da == db:
            exact += 1
        ratio.append(len(set(da) & set(db)) / max(len(db), 1))
    n = max(len(ratio), 1)
    return {"so_cau": len(a), "so_cau_khop_tuyet_doi": exact, "ty_le_khop": exact / n,
            "TB_ti_le_giao": sum(ratio) / n, "cau_thieu_trong_ref": missing}


def cmplogit(got: dict, ref: dict, tol: float = 1e-2, require_full: bool = False) -> dict:
    """So logit theo (qid, doc_id) cho các doc có ở cả hai; exit-code quyết định bởi --require-full."""
    diffs, n_match, n_absent = [], 0, 0
    max_pair = None
    for q, rm in ref.items():
        gh = hit_map(got.get(q, []))
        for d, sc in rm.items():
            h = gh.get(int(d))
            if h is None or h[2] is None:
                n_absent += 1
                continue
            n_match += 1
            df = abs(float(h[2]) - float(sc))
            diffs.append(df)
            if max_pair is None or df > max_pair[2]:
                max_pair = [q, int(d), df]
    rep = {"tol": tol, "so_cap_khop": n_match, "so_cap_thieu_trong_got": n_absent,
           "max_sai": max(diffs) if diffs else None,
           "TB_sai": sum(diffs) / len(diffs) if diffs else None,
           "cap_sai_nhat": max_pair,
           "ok": bool(diffs) and max(diffs) <= tol and (n_absent == 0 or not require_full)}
    return rep


def cmd_pairs(a) -> int:
    pool = load_cache(a.pool)
    p2 = load_cache(a.pool2)
    new = load_cache(a.new)
    idx_caches = [p2] + [load_cache(p) for p in a.idx_from]
    idx_of, idx_bad = collect_idx(idx_caches)
    known = {q: {**hit_map(p2.get(q, [])), **hit_map(pool.get(q, []))} for q in set(pool) | set(p2)}
    cands_m, cands_a, rep = missing_pairs(new, known, idx_of)
    rep["doc_lech_idx"] = idx_bad[:20]
    rep["so_doc_lech_idx"] = len(idx_bad)
    sp, sr = sanity_pairs(pool, p2, a.sanity_q, a.sanity_per)
    rep["sanity_cap"] = sum(len(v) for v in sp.values())
    Path(a.out_cands).write_text(json.dumps(cands_m), encoding="utf-8")
    Path(a.out_new_all).write_text(json.dumps(cands_a), encoding="utf-8")
    Path(a.out_sanity).write_text(json.dumps(sp), encoding="utf-8")
    Path(a.out_sanity_ref).write_text(json.dumps(sr), encoding="utf-8")
    rep["ghi"] = [a.out_cands, a.out_new_all, a.out_sanity, a.out_sanity_ref]
    txt = json.dumps(rep, ensure_ascii=False)
    print(txt, flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")
    return 0 if not idx_bad and not rep["so_thieu_idx"] else 3


def cmd_pool3(a) -> int:
    pool = load_cache(a.pool)
    p2 = load_cache(a.pool2)
    new = load_cache(a.new)
    rerank = load_cache(a.rerank) if a.rerank else {}
    idx_caches = [p2, pool] + [load_cache(p) for p in a.idx_from]
    idx_of, idx_bad = collect_idx(idx_caches)
    known = {q: {**hit_map(p2.get(q, [])), **hit_map(pool.get(q, []))} for q in set(pool) | set(p2)}
    out, rep = build_pool3(pool, new, known, rerank, idx_of)
    rep["doc_lech_idx"] = idx_bad[:20]
    rep["so_thieu_logit_cho_new"] = rep.pop("new_thieu_logit_None")
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    rep["ghi"] = a.out
    txt = json.dumps(rep, ensure_ascii=False)
    print(txt, flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")
    return 0


def cmd_extend(a) -> int:
    base = load_cache(a.base)
    new_scores = load_cache(a.new_scores)
    out, rep = extend_source(base, new_scores)
    Path(a.out).write_text(json.dumps(out), encoding="utf-8")
    rep["ghi"] = a.out
    rep["base"] = a.base
    txt = json.dumps(rep, ensure_ascii=False)
    print(txt, flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")
    return 0


def cmd_doclist(a) -> int:
    pool3 = load_cache(a.pool3)
    rank = load_cache(a.rank)
    new_ids = {str(q): {int(d) for d in v}
               for q, v in json.loads(Path(a.new).read_text(encoding="utf-8")).items()}
    meta_raw = json.loads(Path(a.meta).read_text(encoding="utf-8"))
    meta = {str(q): {int(r["doc_id"]): [s[0] for s in r["sources"]] for r in rows}
            for q, rows in meta_raw.items()}
    out, rep = doclist(pool3, rank, new_ids, meta, a.thr, a.lo, a.hi)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    rep["ghi"] = a.out
    txt = json.dumps(rep, ensure_ascii=False)
    print(txt, flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")
    return 0


def cmd_cmp(a) -> int:
    rep = cmp_cache(load_cache(a.a), load_cache(a.b), a.top)
    txt = json.dumps(rep, ensure_ascii=False)
    print(txt, flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")
    return 0 if rep["ty_le_khop"] >= a.min_ratio else 3


def cmd_cmpsub(a) -> int:
    rep = cmp_sub(json.loads(Path(a.a).read_text(encoding="utf-8")),
                  json.loads(Path(a.b).read_text(encoding="utf-8")))
    txt = json.dumps(rep, ensure_ascii=False)
    print(txt, flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")
    return 0 if rep["ty_le_khop"] >= a.min_ratio else 3


def cmd_cmplogit(a) -> int:
    rep = cmplogit(load_cache(a.got), load_cache(a.ref), a.tol, a.require_full)
    txt = json.dumps(rep, ensure_ascii=False)
    print(txt, flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")
    return 0 if rep["ok"] else 3


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pairs")
    p.add_argument("--pool", required=True)
    p.add_argument("--pool2", required=True)
    p.add_argument("--new", required=True, help="new_cands.json {qid: [doc_id]}")
    p.add_argument("--idx-from", nargs="*", default=[], help="cache nguồn để lấy idx")
    p.add_argument("--out-cands", required=True)
    p.add_argument("--out-new-all", required=True)
    p.add_argument("--out-sanity", required=True)
    p.add_argument("--out-sanity-ref", required=True)
    p.add_argument("--sanity-q", type=int, default=N_SANITY_Q)
    p.add_argument("--sanity-per", type=int, default=N_SANITY_PER)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_pairs)
    p = sp.add_parser("pool3")
    p.add_argument("--pool", required=True)
    p.add_argument("--pool2", required=True)
    p.add_argument("--new", required=True)
    p.add_argument("--rerank", default="", help="rerank_p3.json {qid: [[doc_id, idx, logit], ...]}")
    p.add_argument("--idx-from", nargs="*", default=[])
    p.add_argument("--out", required=True)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_pool3)
    p = sp.add_parser("extend")
    p.add_argument("--base", required=True, help="cache nguồn gốc (giữ nguyên vị trí)")
    p.add_argument("--new-scores", required=True, help="cache điểm NEW (nối đuôi)")
    p.add_argument("--out", required=True)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_extend)
    p = sp.add_parser("doclist")
    p.add_argument("--pool3", required=True)
    p.add_argument("--rank", required=True, help="cache thứ hạng (fhyde3)")
    p.add_argument("--new", required=True)
    p.add_argument("--meta", required=True, help="new_cands_meta.json")
    p.add_argument("--thr", type=float, required=True)
    p.add_argument("--lo", type=int, default=40)
    p.add_argument("--hi", type=int, default=400)
    p.add_argument("--out", required=True)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_doclist)
    p = sp.add_parser("cmp")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--top", type=int, default=TOP_CMP)
    p.add_argument("--min-ratio", type=float, default=0.99)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_cmp)
    p = sp.add_parser("cmpsub")
    p.add_argument("--a", required=True)
    p.add_argument("--b", required=True)
    p.add_argument("--min-ratio", type=float, default=0.99)
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_cmpsub)
    p = sp.add_parser("cmplogit")
    p.add_argument("--got", required=True)
    p.add_argument("--ref", required=True)
    p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument("--require-full", action="store_true")
    p.add_argument("--report", default="")
    p.set_defaults(fn=cmd_cmplogit)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
