"""cx110: submission_cx110cov - nhu cx93cov nhung nhieu chunk hon: k'_q = min(k_q + delta, |C_q|).

Giong cx93cov (cx93.py): cung doc nen (relevant_docs cua submission_split, 0 doc them), cung ung vien
chunk_dw_u900_ext.json.zst, cung s = zc + 4*zd (quy uoc diag14.order_q), cung thuong b = 1,0 cho
chunk s cao nhat cua moi doc (cx93.apply_bonus) - chi khac so chunk moi cau:
  k_q   = so relevant_chunks cua cau trong submission_c120in.json (TB 92,8) - nhu cx93cov
  |C_q| = so ung vien kha dung sau loc doc nen + bo trung nguyen van (len cx93.filter_base)
  k'_q  = min(k_q + delta, |C_q|); chon delta nguyen nho nhat de TB k'_q >= --target-mean (110,0).
  Neu zip LZMA vuot 95.000.000 B thi giam delta toi khi va: check-zip ghi next_delta.txt
  (--delta N) cho .bat chay lai run; moi lan thu deu luu vao delta.thu cua report.

kiem-tai-lap  doi chon delta = 0 voi submission_cx93cov.json (diag15) phai trung Y NGUYEN (bao ti le
              khop cau + vi tri; khong 100% thi exit 2 truoc khi dung tiep). Kiem them tien to: k'_q
              chunk dau cua cx110cov chua dung k_q chunk cua cx93cov lam tien to (thu tu chon giong).
run           kiem-tai-lap + tinh delta + dung submission_cx110cov.json + report.json (TB/min/max k',
              cau chan boi |C_q|, chunk them: so/cau, hang doc trong split, TB logit, TB s,
              nguon ung vien moi docwin_ext vs cu, doc/cau, hist (k_q, |C_q|) cho check-zip).
check-zip     doc lai zip (zipfile + json.load) == JSON goc, zip <= ZIP_LIMIT, cap nhat report
              (json/zip bytes, sha256 + sha256 rut gon). Exit: 0 = vua + dung; 3 = vuot gioi han
              (ghi next_delta.txt); 4 = vuot nhung delta da 0; 1 = loi/noi dung khac.

Dung:
  python scripts/cx110.py run --cands docwin_ext\\chunk_dw_u900_ext.json.zst
      --split diag12\\submission_split.json --fused diag10\\cache_rrf3d2.json
      --ref-c120in diag14\\submission_c120in.json --pool cache_pool_rr.json
      --ref-cx93cov diag15\\submission_cx93cov.json --dq cache_docwin_ext.json --outdir diag16
  python scripts/cx110.py check-zip --json diag16\\submission_cx110cov.json
      --zip diag16\\submission_cx110cov.zip --report diag16\\report.json
"""
import argparse
import hashlib
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from cx93 import B_COV, BAND_NAMES, WD, ZIP_LIMIT, apply_bonus, filter_base, load_inputs
from diag14 import order_q, strip_meta
from diag14 import verify_zip as verify_zip_bytes
from dwx_cands import band_of

TARGET_MEAN = 110.0


def norm_pairs(sel: list) -> list:
    """[(str(doc_id), chunk_text)] - so sanh khong phu thuoc kieu doc_id."""
    return [(str(c["doc_id"]), c["chunk_text"]) for c in sel]


def kprime(k_q: int, cap: int, delta: int) -> int:
    return min(k_q + delta, cap)


def hist_from_pairs(pairs: list) -> list:
    """[[k_q, |C_q|, so_cau], ...] - hist dem cho check-zip uoc delta tiep theo."""
    return [[k, cap, c] for (k, cap), c in sorted(Counter(pairs).items())]


def total_chunks(hist: list, delta: int) -> int:
    return sum(kprime(k, cap, delta) * c for k, cap, c in hist)


def choose_delta(pairs: list, target: float) -> tuple[int, bool]:
    """(delta, dat_target): delta nguyen nho nhat >= 0 de TB min(k_q+delta, |C_q|) >= target.

    Khong dat duoc (TB |C_q| < target): delta toi da hoa TB (tat ca cau deu bi chan), dat_target=False.
    """
    n = max(len(pairs), 1)
    if sum(cap for _k, cap in pairs) / n < target:
        return max(max((cap - k for k, cap in pairs), default=0), 0), False
    lo, hi = 0, max(max((cap - k for k, cap in pairs), default=0), 0)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum(kprime(k, cap, mid) for k, cap in pairs) / n >= target:
            hi = mid
        else:
            lo = mid + 1
    return lo, True


def check_repro(prep: dict, ref93: dict) -> dict:
    """Chon delta = 0 (k = min(k_q, |C_q|)) tren ung vien ext, so Y NGUYEN voi submission_cx93cov.

    Tra {cau, cau_khop, ty_le_cau_khop, vi_tri_khop, ty_le_vi_tri_chunk_khop, ok, vi_du_lech}.
    """
    n = n_ok = n_pos = n_pos_tot = 0
    vi_du = []
    for q, want in ref93.items():
        p = prep.get(q)
        got = strip_meta(p["ob"], min(p["kq"], p["cap"])) if p else []
        w, g = norm_pairs(want), norm_pairs(got)
        n += 1
        ok = g == w
        n_ok += ok
        for x, y in zip(g, w):
            n_pos_tot += 1
            n_pos += x == y
        n_pos_tot += abs(len(g) - len(w))
        if not ok and len(vi_du) < 5:
            i = next((j for j in range(max(len(g), len(w))) if g[j:j + 1] != w[j:j + 1]), None)
            vi_du.append({"id": q, "k_q": len(w), "got_n": len(g), "vi_tri_lech": i,
                          "got": g[i] if i is not None and i < len(g) else None,
                          "want": w[i] if i is not None and i < len(w) else None})
    return {"cau": n, "cau_khop": n_ok, "ty_le_cau_khop": n_ok / max(n, 1),
            "vi_tri_khop": n_pos, "ty_le_vi_tri_chunk_khop": n_pos / max(n_pos_tot, 1),
            "ok": n_ok == n and n == len(prep), "vi_du_lech": vi_du}


def build_sub(split: list, prep: dict, ref93: dict, dq: dict, rank: dict, delta: int,
              base: dict) -> tuple[list, dict]:
    """Dung submission_cx110cov: k'_q = min(k_q + delta, |C_q|) chunk dau cua order da thuong b.

    Tra (sub, stats). Tien to cx93cov kiem theo ref93 (file da luu). Chunk them = cac vi tri sau
    chunk cx93cov (k93 = min(k_q, |C_q|)) - phan tich theo hang doc trong split, logit, s,
    nguon (doc thuoc D_q cache_docwin_ext = ung vien moi docwin_ext; con lai = ung vien cu).
    """
    sub = []
    kps, n_doc_q, n_them_q = [], [], []
    bands = {name: 0 for name in BAND_NAMES}
    sum_logit = sum_s = 0.0
    n_new = n_old = n_them = blocked = n_prefix_ok = 0
    for it in split:
        q = str(it["id"])
        p = prep[q]
        kq, cap, ob = p["kq"], p["cap"], p["ob"]
        kp = kprime(kq, cap, delta)
        k93 = min(kq, cap)
        sel = strip_meta(ob, kp)
        extra = ob[k93:kp]
        w = norm_pairs(ref93.get(q, []))
        g = norm_pairs(sel)
        n_prefix_ok += g[:len(w)] == w
        blocked += cap < kq + delta
        kps.append(kp)
        n_doc_q.append(len({str(c["doc_id"]) for c in sel}))
        n_them_q.append(len(extra))
        dqs = dq.get(q, set())
        for c in extra:
            d = str(c["doc_id"])
            assert d in rank[q], f"chunk them ngoai doc nen: {q} {d}"
            bands[band_of(rank[q][d])] += 1
            sum_logit += c["logit"]
            sum_s += c["s"]
            if d in dqs:
                n_new += 1
            else:
                n_old += 1
        n_them += len(extra)
        for c in sel:
            assert str(c["doc_id"]) in base[q]
        sub.append({"id": it["id"], "relevant_docs": list(it["relevant_docs"]),
                    "relevant_chunks": sel})
    n = max(len(sub), 1)
    st = {"kprime": {"TB": sum(kps) / n, "min": min(kps, default=0), "max": max(kps, default=0),
                     "tong": sum(kps)},
          "cau_chan_boi_C_q": blocked,
          "cau_tien_to_cx93cov_ok": n_prefix_ok,
          "TB_chunk_cau": sum(kps) / n, "min_chunk": min(kps, default=0), "max_chunk": max(kps, default=0),
          "tong_chunk": sum(kps),
          "TB_doc_khac_nhau_co_chunk_cau": sum(n_doc_q) / n,
          "TB_doc_cau": sum(len(it["relevant_docs"]) for it in sub) / n,
          "TB_doc_them_cau": 0.0,
          "TB_chunk_them_cau": sum(n_them_q) / n,
          "chunk_them": {
              "so": n_them,
              "TB_moi_cau": n_them / n,
              "phan_bang_hang_doc": {name: {"so_chunk": bands[name], "TB_moi_cau": bands[name] / n}
                                     for name in BAND_NAMES},
              "TB_logit": sum_logit / n_them if n_them else None,
              "TB_s": sum_s / n_them if n_them else None,
              "nguon": {"moi_docwin_ext": n_new, "cu": n_old,
                        "ti_le_moi": n_new / n_them if n_them else None},
          }}
    return sub, st


def run(cands: str, split_path: str, fused_path: str, ref_path: str, pool_path: str,
        ref_cx93cov_path: str, dq_path: str, outdir: str, target_mean: float = TARGET_MEAN,
        delta: int | None = None, wd: float = WD, b_cov: float = B_COV,
        report_path: str = "") -> dict:
    t0 = time.time()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rr, split, fu, ref_chunks, base, rank, _top = load_inputs(cands, split_path, fused_path, ref_path, pool_path)
    ref93 = {str(it["id"]): it["relevant_chunks"]
             for it in json.loads(Path(ref_cx93cov_path).read_text(encoding="utf-8"))}
    dq = {str(q): {str(h[0]) for h in (hits or [])}
          for q, hits in json.loads(Path(dq_path).read_text(encoding="utf-8")).items()}
    print(f"nap xong {time.time() - t0:.0f}s: {len(rr)} cau ung vien, {len(split)} cau", flush=True)
    prep, pairs = {}, []
    for it in split:
        q = str(it["id"])
        cin = filter_base(order_q(rr.get(q, []), fu.get(q, []), wd), base[q])
        kq = len(ref_chunks.get(q, []))
        prep[q] = {"kq": kq, "cap": len(cin), "ob": apply_bonus(cin, b_cov)}
        pairs.append((kq, len(cin)))
    del rr, fu
    rp = Path(report_path) if report_path else out / "report.json"
    thu = []
    if rp.exists():
        try:
            thu = json.loads(rp.read_text(encoding="utf-8")).get("delta", {}).get("thu", []) or []
        except (json.JSONDecodeError, OSError):
            thu = []
    repro = check_repro(prep, ref93)
    print(f"tai lap delta=0: {json.dumps(repro, ensure_ascii=False)}", flush=True)
    if not repro["ok"]:
        rp.write_text(json.dumps({"quy_uoc": "cx110", "tai_lap_delta0": repro}, ensure_ascii=False),
                      encoding="utf-8")
        print(f"KHONG tai lap duoc cx93cov, dung lai; ghi {rp}", flush=True)
        raise SystemExit(2)
    delta_target, dat = choose_delta(pairs, target_mean)
    d_use = delta_target if delta is None else int(delta)
    sub, st = build_sub(split, prep, ref93, dq, rank, d_use, base)
    del prep
    fp = out / "submission_cx110cov.json"
    with fp.open("w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False)
    st["json_bytes"] = fp.stat().st_size
    ly_do = (f"mac dinh: delta nho nhat de TB k' >= {target_mean}" if delta is None
             else f"ghi nhan --delta {d_use} (target {delta_target}, co the da giam vi zip)")
    report = {"quy_uoc": {
                  "giong": "cx93cov: doc nen = relevant_docs split (0 doc them), ung vien "
                           "chunk_dw_u900_ext.json.zst, s = zc + 4*zd (diag14.order_q), b = 1,0 (apply_bonus)",
                  "khac": "k'_q = min(k_q + delta, |C_q|); |C_q| = ung vien sau loc doc nen + bo trung",
                  "k_q": "so relevant_chunks cua cau trong submission_c120in.json",
                  "chunk_them": "vi tri sau chunk cx93cov (min(k_q, |C_q|)) cua cung thu tu chon",
                  "nguon_moi": "doc D_q cua cache_docwin_ext.json = ung vien moi docwin_ext; con lai = cu",
                  "zip": "zip_submission.py --method lzma, entry submission.json"},
              "wd": wd, "b_cov": b_cov, "target_mean": target_mean, "zip_limit": ZIP_LIMIT,
              "cau": len(split), "k_q_TB": sum(k for k, _ in pairs) / max(len(pairs), 1),
              "tai_lap_delta0": repro,
              "delta": {"target": delta_target, "dung": d_use, "dat_target": dat, "ly_do": ly_do,
                        "thu": thu},
              "kprime": st["kprime"], "cau_chan_boi_C_q": st["cau_chan_boi_C_q"],
              "cx110cov": st,
              "hist_kcap": hist_from_pairs(pairs),
              "thoi_gian_giay": round(time.time() - t0, 1)}
    rp.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"delta {delta_target} -> dung {d_use}: {json.dumps(st, ensure_ascii=False)}", flush=True)
    print(f"ghi {rp}; tong {report['thoi_gian_giay']}s", flush=True)
    return report


def suggest_next_delta(hist: list, delta_now: int, n_chunks: int, zip_bytes: int, limit: int,
                       marg: float | None = None) -> int:
    """delta < delta_now sao cho tong chunk giam du de zip <= limit (uoc luong tuyen tinh)."""
    if delta_now <= 0:
        return 0
    m = marg if (marg is not None and marg > 0) else zip_bytes / max(n_chunks, 1)
    drop = max(1, math.ceil((zip_bytes - limit) / m))
    want = n_chunks - drop
    if total_chunks(hist, 0) > want:
        return 0
    lo, hi = 0, delta_now - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if total_chunks(hist, mid) <= want:
            lo = mid
        else:
            hi = mid - 1
    return lo


def check_zip(json_path: str, zip_path: str, report_path: str) -> int:
    """Doc lai zip == JSON goc + zip <= ZIP_LIMIT; cap nhat report. Tra exit code (0/1/3/4)."""
    p, zp = Path(json_path), Path(zip_path)
    st = dict(verify_zip_bytes(str(p), str(zp)))
    zb, jb = zp.stat().st_size, p.stat().st_size
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    duoi = zb <= ZIP_LIMIT
    ok_noi_dung = bool(st["ok"])
    rp = Path(report_path)
    report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    d_used = report.get("delta", {}).get("dung")
    n_chunks = report.get("cx110cov", {}).get("tong_chunk", 0)
    attempt = {"delta": d_used, "json_bytes": jb, "zip_bytes": zb, "tong_chunk": n_chunks,
               "duoi_gioi_han": duoi}
    marg = None
    prev = [t for t in report.get("delta", {}).get("thu", []) or [] if t.get("tong_chunk")]
    if prev and prev[-1]["tong_chunk"] != n_chunks:
        marg = (zb - prev[-1]["zip_bytes"]) / (n_chunks - prev[-1]["tong_chunk"])
    report.setdefault("delta", {}).setdefault("thu", []).append(attempt)
    next_d = None
    code = 0
    if ok_noi_dung and not duoi:
        if d_used is None or d_used <= 0:
            code = 4
        else:
            next_d = suggest_next_delta(report.get("hist_kcap", []), int(d_used), n_chunks, zb, ZIP_LIMIT, marg)
            code = 3
    elif not ok_noi_dung:
        code = 1
    report["zip"] = {"entry": st["entry"], "bang_goc_bytes": st["bang_goc_bytes"],
                     "bang_goc_json": st["bang_goc_json"], "json_bytes": jb, "zip_bytes": zb,
                     "duoi_gioi_han": duoi, "sha256": sha, "sha256_rut_gon": sha[:16],
                     "ok": code == 0}
    rp.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    if next_d is not None:
        (p.parent / "next_delta.txt").write_text(f"--delta {next_d}", encoding="utf-8")
    print(json.dumps({**report["zip"], "next_delta": next_d}, ensure_ascii=False), flush=True)
    return code


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    rn = sp.add_parser("run")
    rn.add_argument("--cands", required=True, help="ung vien chunk (.json/.json.zst)")
    rn.add_argument("--split", required=True, help="submission_split.json (doc nen)")
    rn.add_argument("--fused", required=True, help="cache_rrf3d2.json (nguon zd)")
    rn.add_argument("--ref-c120in", required=True, help="submission_c120in.json (k_q)")
    rn.add_argument("--pool", required=True, help="cache_pool_rr.json")
    rn.add_argument("--ref-cx93cov", required=True, help="submission_cx93cov.json (kiem tai lap delta=0)")
    rn.add_argument("--dq", required=True, help="cache_docwin_ext.json (D_q = ung vien moi)")
    rn.add_argument("--outdir", required=True)
    rn.add_argument("--report", default="", help="JSON bao cao (mac dinh <outdir>/report.json)")
    rn.add_argument("--target-mean", type=float, default=TARGET_MEAN)
    rn.add_argument("--delta", type=int, default=None, help="ghi nhan delta (mac dinh tinh tu target)")
    rn.add_argument("--wd", type=float, default=WD)
    rn.add_argument("--b-cov", type=float, default=B_COV)
    cz = sp.add_parser("check-zip")
    cz.add_argument("--json", required=True)
    cz.add_argument("--zip", required=True)
    cz.add_argument("--report", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a.cands, a.split, a.fused, a.ref_c120in, a.pool, a.ref_cx93cov, a.dq, a.outdir,
            a.target_mean, a.delta, a.wd, a.b_cov, a.report)
        raise SystemExit(0)
    raise SystemExit(check_zip(a.json, a.zip, a.report))


if __name__ == "__main__":
    main()
