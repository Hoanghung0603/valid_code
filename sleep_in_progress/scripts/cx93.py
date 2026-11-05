"""cx93: 2 bài nộp nhiều chunk từ ứng viên mở rộng chunk_dw_u900_ext (CPU b660), KHÔNG nộp.

Quy ước điểm (giống diag14.order_q / chunk_docfuse --wd 4):
  zc = z-score logit chunk trên TOÀN BỘ ứng viên của câu trong file ứng viên (kể cả dòng
       doc_id không thuộc doc nền) — đúng quy ước diag14 (order_q tính zc trên mọi dòng,
       khác "zc trên C_q"); zd = z-score điểm fused của doc trong cache_rrf3d2 (doc không
       có -> zd nhỏ nhất của câu); s = zc + 4*zd.
  Bỏ trùng nguyên văn GIỮ DÒNG ĐI TRƯỚC trong thứ tự s giảm dần TRƯỚC khi lọc doc nền
  (quy ước diag14; khác "giữ dòng logit cao hơn" khi cùng text ở nhiều doc — repro đo tần suất).
  C_q = dòng của order_q có doc_id trong relevant_docs của submission_split (doc nền), lấy đủ.
  k_q = số relevant_chunks của câu trong submission_c120in.json (ngân sách từng câu như #5508).

cx93in   k_q chunk đầu của C_q theo s.
cx93cov  cộng thưởng b=1,0 vào chunk s cao nhất của mỗi doc (chunk đầu tiên của doc),
         sắp theo s' = s + b rồi lấy k_q chunk.

repro    cách chọn cx93in trên ứng viên CŨ chunk_dw_u900.json.zst phải trùng y nguyên
         relevant_chunks của submission_c120in.json (exit 2 nếu không 100%). Đo thêm biến thể
         "C_q literal" (lọc doc nền -> bỏ trùng giữ dòng logit cao hơn -> zc trên C_q) để ghi rõ
         chênh lệch, cùng thống kê text trùng nguyên văn.

run      dựng submission_cx93in.json + submission_cx93cov.json vào --outdir + report.json
         (TB chunk/doc/câu, doc có chunk/câu, chunk mới/câu, lệch c120in/câu, phân bố hạng doc,
         b-quét TB doc có chunk/câu với b ∈ {0; 0,5; 1,0; 2,0; ∞}).

verify-zip  đọc lại zip (zipfile + json.load) so JSON gốc + kiểm zip ≤ 95.000.000 B.

Dùng:
  python scripts/cx93.py repro --cands docwin\\chunk_dw_u900.json.zst --split diag12\\submission_split.json
      --fused diag10\\cache_rrf3d2.json --ref-c120in diag14\\submission_c120in.json --report repro_report.json
  python scripts/cx93.py run --cands docwin_ext\\chunk_dw_u900_ext.json.zst --split ... --fused ...
      --ref-c120in ... --pool cache_pool_rr.json --outdir diag15
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from chunk_docfuse import doc_z, zscores
from chunks_from_rerank import load_rerank
from diag14 import order_q, strip_meta
from diag14 import verify_zip as verify_zip_bytes
from dwx_cands import band_of

WD = 4.0
B_COV = 1.0
B_VALUES = (0.0, 0.5, 1.0, 2.0, float("inf"))
ZIP_LIMIT = 95_000_000
BAND_NAMES = ("1-30", "31-60", "61-100", "101+")


def filter_base(order: list, base_docs: set) -> list:
    """Giữ chunk có doc_id trong doc nền (so theo str như diag14.build_sub c120in)."""
    return [c for c in order if str(c["doc_id"]) in base_docs]


def apply_bonus(order: list, b: float) -> list:
    """Cộng thưởng b vào chunk s cao nhất của mỗi doc (chunk đầu tiên trong thứ tự s), sắp s' giảm dần.

    order phải đang sắp theo s giảm dần (như đầu ra filter_base). b = inf: mọi chunk đầu doc
    đứng trước (vẫn xếp theo s giảm dần trong từng nhóm). Trả danh sách mới, không đổi order.
    """
    seen, keys = set(), []
    for c in order:
        d = str(c["doc_id"])
        keys.append(c["s"] + (b if d not in seen else 0.0))
        seen.add(d)
    return [order[i] for i in sorted(range(len(order)), key=lambda i: -keys[i])]


def select_q(rows: list, fused_hits: list, base_docs: set, k: int, b: float = 0.0, wd: float = WD) -> list[dict]:
    """k_q chunk của C_q theo s' = s + thưởng b (b = 0: k đầu theo s). Quy ước diag14."""
    order = filter_base(order_q(rows, fused_hits, wd), base_docs)
    if b:
        order = apply_bonus(order, b)
    return strip_meta(order, k)


def select_q_literal(rows: list, fused_hits: list, base_docs: set, k: int, wd: float = WD) -> list[dict]:
    """Biến thể đọc chữ 'C_q': lọc doc nền trước, bỏ trùng giữ dòng logit cao hơn, zc trên C_q."""
    keep, pos = [], {}
    for r in rows:
        if str(r[0]) not in base_docs:
            continue
        t = r[4]
        if t not in pos:
            pos[t] = len(keep)
            keep.append(r)
        elif float(r[3]) > float(keep[pos[t]][3]):
            keep[pos[t]] = r
    zc = zscores([float(r[3]) for r in keep])
    zdm = doc_z(fused_hits)
    zmin = min(zdm.values()) if zdm else 0.0
    scored = [(c + wd * zdm.get(str(r[0]), zmin), r) for c, r in zip(zc, keep)]
    return [{"doc_id": r[0], "chunk_text": r[4]} for _, r in sorted(scored, key=lambda x: -x[0])][:k]


def dup_text_stats(rows: list, order: list) -> dict:
    """Thống kê dòng trùng nguyên văn của 1 câu: text nhiều dòng, khác doc, lệch luật giữ dòng.

    Luật s (diag14) giữ dòng đi trước trong thứ tự s giảm dần = dòng đầu của text trong order
    (order_q đã bỏ trùng); luật logit giữ dòng logit cao hơn (hòa: dòng đứng trước).
    """
    by = {}
    for r in rows:
        by.setdefault(r[4], []).append(r)
    winner_s = {c["chunk_text"]: c for c in order}
    out = {"text_nhieu_dong": 0, "text_khac_doc": 0, "lech_luat_giu_dong": 0}
    for t, rs in by.items():
        if len(rs) < 2:
            continue
        out["text_nhieu_dong"] += 1
        if len({str(r[0]) for r in rs}) > 1:
            out["text_khac_doc"] += 1
            best = max(rs, key=lambda r: float(r[3]))
            if str(winner_s[t]["doc_id"]) != str(best[0]):
                out["lech_luat_giu_dong"] += 1
    return out


def repro_check(rr: dict, fu: dict, base: dict, ref_chunks: dict, wd: float = WD) -> dict:
    """Chọn cx93in trên ứng viên cho trước, so Y NGUYÊN với relevant_chunks của c120in.

    Trả {cau, cau_khop, ty_le_cau_khop, vi_tri_khop, ty_le_vi_tri_chunk_khop, bien_the_C_q_literal,
    trung_nguyen_van, cau_chenh_k_q, ok, vi_du_lech}.
    """
    n = n_ok = n_lit = n_pos = n_pos_tot = n_chenh_k = 0
    vi_du = []
    dup = {"text_nhieu_dong": 0, "text_khac_doc": 0, "lech_luat_giu_dong": 0}
    for q, want in ref_chunks.items():
        rows, hits = rr.get(q, []), fu.get(q, [])
        b = base[q]
        order = order_q(rows, hits, wd)
        got = strip_meta(filter_base(order, b), len(want))
        got_lit = select_q_literal(rows, hits, b, len(want), wd)
        for key, v in dup_text_stats(rows, order).items():
            dup[key] += v
        n += 1
        ok = got == want
        n_ok += ok
        n_lit += got_lit == want
        n_chenh_k += len(filter_base(order, b)) != len(want)
        for x, y in zip(got, want):
            n_pos_tot += 1
            n_pos += x == y
        n_pos_tot += abs(len(got) - len(want))
        if not ok and len(vi_du) < 5:
            i = next((j for j in range(max(len(got), len(want))) if got[j:j + 1] != want[j:j + 1]), None)
            vi_du.append({"id": q, "k_q": len(want), "got_n": len(got), "vi_tri_lech": i,
                          "got": got[i] if i is not None and i < len(got) else None,
                          "want": want[i] if i is not None and i < len(want) else None})
    return {"cau": n, "cau_khop": n_ok, "ty_le_cau_khop": n_ok / max(n, 1),
            "vi_tri_khop": n_pos, "ty_le_vi_tri_chunk_khop": n_pos / max(n_pos_tot, 1),
            "bien_the_C_q_literal": {"cau_khop": n_lit, "ty_le_cau_khop": n_lit / max(n, 1)},
            "trung_nguyen_van": dup, "cau_C_q_khac_k_q": n_chenh_k,
            "ok": n_ok == n, "vi_du_lech": vi_du}


def pair_set(sel: list) -> set:
    return {(str(c["doc_id"]), c["chunk_text"]) for c in sel}


def stats_q(sel: list, want: list, rank_q: dict, top100_q: set, k: int) -> dict:
    """Thống kê 1 câu cho 1 bài: số chunk/doc, chunk mới, lệch so c120in, hạng doc trong split."""
    pairs, refp = pair_set(sel), pair_set(want)
    bands = {name: 0 for name in BAND_NAMES}
    n_moi = 0
    for c in sel:
        d = str(c["doc_id"])
        bands[band_of(rank_q[d])] += 1
        if d not in top100_q:
            n_moi += 1
    return {"n_chunk": len(sel), "n_doc": len({str(c["doc_id"]) for c in sel}), "n_moi": n_moi,
            "n_khac": len(pairs - refp), "n_mat": len(refp - pairs),
            "n_thieu_k": max(0, k - len(sel)), "bands": bands}


def aggregate(per_q: list) -> dict:
    n = max(len(per_q), 1)
    bands = {}
    for name in BAND_NAMES:
        total = sum(p["bands"][name] for p in per_q)
        bands[name] = {"so_chunk": total, "TB_moi_cau": total / n}
    return {"TB_chunk_cau": sum(p["n_chunk"] for p in per_q) / n,
            "min_chunk": min((p["n_chunk"] for p in per_q), default=0),
            "max_chunk": max((p["n_chunk"] for p in per_q), default=0),
            "TB_doc_khac_nhau_co_chunk_cau": sum(p["n_doc"] for p in per_q) / n,
            "TB_chunk_moi_cau": sum(p["n_moi"] for p in per_q) / n,
            "TB_chunk_khac_c120in_cau": sum(p["n_khac"] for p in per_q) / n,
            "TB_thieu_c120in_cau": sum(p["n_mat"] for p in per_q) / n,
            "TB_lech_c120in_cau": sum(p["n_khac"] + p["n_mat"] for p in per_q) / n,
            "cau_khac_c120in": sum(1 for p in per_q if p["n_khac"] or p["n_mat"]),
            "cau_thieu_k_q": sum(1 for p in per_q if p["n_thieu_k"]),
            "phan_bang_hang_doc": bands}


def build_subs(split: list, rr: dict, fu: dict, ref_chunks: dict, rank: dict, top100: dict,
               wd: float = WD, b_cov: float = B_COV, b_values: tuple = B_VALUES) -> tuple[list, list, dict]:
    """Dựng cx93in + cx93cov từ ứng viên rr. Trả (sub_in, sub_cov, stats).

    ref_chunks[q] = relevant_chunks của c120in (cho k_q + so lệch); rank[q] = {str(doc): hạng trong
    relevant_docs của split}; top100[q] = set(str(doc)) top-100 bge (cache_pool_rr) để phân loại chunk mới.
    """
    sub_in, sub_cov = [], []
    acc = {"cx93in": [], "cx93cov": []}
    sweep = {str(b): [] for b in b_values}
    dup = {"text_nhieu_dong": 0, "text_khac_doc": 0, "lech_luat_giu_dong": 0}
    kq_sum = 0
    for it in split:
        q = str(it["id"])
        base = {str(d) for d in it["relevant_docs"]}
        rows, hits = rr.get(q, []), fu.get(q, [])
        want = ref_chunks.get(q, [])
        k = len(want)
        kq_sum += k
        order = order_q(rows, hits, wd)
        cin = filter_base(order, base)
        for key, v in dup_text_stats(rows, order).items():
            dup[key] += v
        sels = {"cx93in": strip_meta(cin, k), "cx93cov": strip_meta(apply_bonus(cin, b_cov), k)}
        for name, sel in sels.items():
            acc[name].append(stats_q(sel, want, rank[q], top100[q], k))
            for c in sel:
                assert str(c["doc_id"]) in base
        for b in b_values:
            sweep[str(b)].append(len({str(c["doc_id"]) for c in strip_meta(apply_bonus(cin, b), k)}))
        sub_in.append({"id": it["id"], "relevant_docs": list(it["relevant_docs"]),
                       "relevant_chunks": sels["cx93in"]})
        sub_cov.append({"id": it["id"], "relevant_docs": list(it["relevant_docs"]),
                        "relevant_chunks": sels["cx93cov"]})
    n = max(len(split), 1)
    doc_tb = sum(len(it["relevant_docs"]) for it in split) / n
    for name, sub in (("cx93in", sub_in), ("cx93cov", sub_cov)):
        st = aggregate(acc[name])
        st["TB_doc_cau"] = sum(len(it["relevant_docs"]) for it in sub) / n
        st["TB_doc_them_cau"] = 0.0
        assert st["TB_doc_cau"] == doc_tb
        acc[name] = st
    stats = {"cau": len(split), "k_q_TB": kq_sum / n, "bai": acc,
             "b_quet_TB_doc_co_chunk_cau": {b: sum(v) / max(len(v), 1) for b, v in sweep.items()},
             "trung_nguyen_van": dup}
    return sub_in, sub_cov, stats


def verify_zip_full(json_path: str, zip_path: str) -> dict:
    """diag14.verify_zip + kiểm zip ≤ ZIP_LIMIT."""
    st = dict(verify_zip_bytes(json_path, zip_path))
    st["zip_bytes"] = Path(zip_path).stat().st_size
    st["duoi_gioi_han"] = st["zip_bytes"] <= ZIP_LIMIT
    st["ok"] = bool(st["ok"] and st["duoi_gioi_han"])
    return st


def load_inputs(cands: str, split_path: str, fused_path: str, ref_path: str, pool_path: str = "") -> tuple:
    """Nạp ứng viên + split + fused + c120in (+pool). Trả (rr, split, fu, ref_chunks, rank, top100)."""
    rr = {str(q): v for q, v in load_rerank(cands).items()}
    fu = {str(q): v for q, v in json.loads(Path(fused_path).read_text(encoding="utf-8")).items()}
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    ref_chunks = {str(it["id"]): it["relevant_chunks"] for it in ref}
    base = {str(it["id"]): {str(d) for d in it["relevant_docs"]} for it in split}
    rank = {str(it["id"]): {str(d): i + 1 for i, d in enumerate(it["relevant_docs"])} for it in split}
    if pool_path:
        pool = {str(q): v for q, v in json.loads(Path(pool_path).read_text(encoding="utf-8")).items()}
    else:
        pool = {q: [] for q in base}
    top100 = {q: {str(h[0]) for h in (pool.get(q) or [])[:100]} for q in base}
    del ref
    return rr, split, fu, ref_chunks, base, rank, top100


def run(cands: str, split_path: str, fused_path: str, ref_path: str, pool_path: str, outdir: str,
        report_path: str = "", wd: float = WD, b_cov: float = B_COV) -> dict:
    t0 = time.time()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rr, split, fu, ref_chunks, base, rank, top100 = load_inputs(cands, split_path, fused_path, ref_path, pool_path)
    print(f"nap xong {time.time() - t0:.0f}s: {len(rr)} cau ung vien, {len(split)} cau", flush=True)
    sub_in, sub_cov, stats = build_subs(split, rr, fu, ref_chunks, rank, top100, wd, b_cov)
    del rr, fu
    files = {"cx93in": sub_in, "cx93cov": sub_cov}
    for name, sub in files.items():
        fp = out / f"submission_{name}.json"
        with fp.open("w", encoding="utf-8") as f:
            json.dump(sub, f, ensure_ascii=False)
        stats["bai"][name]["json_bytes"] = fp.stat().st_size
        print(f"{name}: {json.dumps(stats['bai'][name], ensure_ascii=False)}", flush=True)
    stats["b_quet"] = stats.pop("b_quet_TB_doc_co_chunk_cau")
    report = {"quy_uoc": {
                  "zc": "z-score logit trên toàn bộ ứng viên của câu trong file (quy ước diag14.order_q, KHÔNG chỉ C_q)",
                  "zd": "z-score điểm fused rrf3d2 của doc, doc không có -> zd nhỏ nhất của câu",
                  "s": "zc + 4*zd; bỏ trùng nguyên văn giữ dòng s cao nhất TRƯỚC khi lọc doc nền (quy ước diag14)",
                  "k_q": "số relevant_chunks của câu trong submission_c120in.json",
                  "doc_nen": "relevant_docs của submission_split, giữ y nguyên (0 doc thêm)",
                  "b_cov": b_cov},
              "wd": wd, "zip_limit": ZIP_LIMIT, **stats,
              "thoi_gian_giay": round(time.time() - t0, 1)}
    rp = Path(report_path) if report_path else out / "report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"b_quet: {json.dumps(report['b_quet'], ensure_ascii=False)}", flush=True)
    print(f"trung nguyen van: {json.dumps(report['trung_nguyen_van'], ensure_ascii=False)}", flush=True)
    print(f"ghi {rp}; tong {report['thoi_gian_giay']}s", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("repro", "run"):
        p = sp.add_parser(name)
        p.add_argument("--cands", required=True, help="file ứng viên chunk (.json/.json.zst)")
        p.add_argument("--split", required=True, help="submission_split.json (doc nền)")
        p.add_argument("--fused", required=True, help="cache_rrf3d2.json (nguồn zd)")
        p.add_argument("--ref-c120in", required=True, help="submission_c120in.json (k_q + so tái lập)")
        p.add_argument("--report", default="", help="JSON báo cáo (mặc định <outdir>/report.json)")
        p.add_argument("--wd", type=float, default=WD)
    sp.choices["repro"].add_argument("--pool", default="", help="cache_pool_rr.json (phân loại chunk mới)")
    sp.choices["run"].add_argument("--pool", required=True, help="cache_pool_rr.json (phân loại chunk mới)")
    sp.choices["run"].add_argument("--outdir", required=True)
    sp.choices["run"].add_argument("--b-cov", type=float, default=B_COV, help="thưởng b của cx93cov")
    vz = sp.add_parser("verify-zip")
    vz.add_argument("--json", required=True)
    vz.add_argument("--zip", required=True)
    a = ap.parse_args()
    if a.cmd == "repro":
        rr, split, fu, ref_chunks, base, _rank, _top = load_inputs(a.cands, a.split, a.fused, a.ref_c120in, a.pool)
        rep = repro_check(rr, fu, base, ref_chunks, a.wd)
        rp = a.report or "repro_report.json"
        Path(rp).write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(rep, ensure_ascii=False), flush=True)
        print(f"ghi {rp}", flush=True)
        raise SystemExit(0 if rep["ok"] else 2)
    if a.cmd == "run":
        run(a.cands, a.split, a.fused, a.ref_c120in, a.pool, a.outdir, a.report, a.wd, a.b_cov)
        raise SystemExit(0)
    st = verify_zip_full(a.json, a.zip)
    print(json.dumps(st, ensure_ascii=False), flush=True)
    raise SystemExit(0 if st["ok"] else 1)


if __name__ == "__main__":
    main()
