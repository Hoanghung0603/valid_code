"""cx130: submission_cx130cov - nhu cx110cov (cx110.py) nhung nhieu chunk hon: k'_q = min(k_q + delta,
|C_q|), chon delta de TB k' >= 130,0; zip LZMA <= 93.000.000 B (chua bien duoi 100MB).

Giong cx110cov: doc nen = relevant_docs cua submission_split (0 doc them), ung vien
chunk_dw_u900_ext.json.zst, s = zc + 4*zd (diag14.order_q), thuong b = 1,0 (cx93.apply_bonus),
k_q = so relevant_chunks cua cau trong submission_c120in.json. Chi khac so chunk moi cau:
  k'_q  = min(k_q + delta, |C_q|); |C_q| = ung vien sau loc doc nen + bo trung nguyen van.
  Chon delta nguyen nho nhat de TB k'_q >= --target-mean (130,0). Neu khong dat duoc (|C_q| chan)
  hoac uoc zip LZMA > --max-zip (93.000.000) thi lay delta lon nhat theo uoc tinh zip, ghi ly_do.
  Uoc kich thuoc zip theo ti le byte JSON/zip cua cx110cov (505.693.628 / 79.380.644) TRUOC khi nen
  that; uoc bang JSON da ghi + chinh sua anchor theo lan build; chi nen LZMA khi uoc duoi ngan sach.
  Neu zip that vuot --max-zip thi check-zip ghi next_delta.txt cho .bat giam delta va nen lai.

kiem-tai-lap  (1) delta = 23 phai trung Y NGUYEN submission_cx110cov.json (1200/1200 cau);
              (2) k'_q chunk dau cua bai moi chua dung chunk cua cx110cov lam tien to (mn cau,
              khi delta >= 23). Khong 100% thi exit 2 truoc khi dung tiep.
run           kiem-tai-lap + chon delta + dung submission_cx130cov.json + report.json (TB/min/max k',
              cau chan boi |C_q|, tien to cx110cov, chunk them vs cx110cov: so/cau, hang doc trong
              split, TB logit, TB s, nguon docwin_ext vs ung vien cu, doc/cau, hist (k_q, |C_q|),
              cac lan uoc zip). Exit: 0 = uoc zip duoi ngan sach; 2 = loi tai lap; 3 = vuot nhung
              con giam duoc (ghi next_delta.txt); 4 = vuot nhung delta da 0.
check-zip     doc lai zip (zipfile + json.load) == JSON goc, zip <= max_zip, cap nhat report
              (json/zip bytes, sha256 + sha256 rut gon). Exit: 0 = vua + dung; 3 = vuot gioi han
              (ghi next_delta.txt); 4 = vuot nhung delta da 0; 1 = loi/noi dung khac.

Dung:
  python scripts/cx130.py run --cands docwin_ext\\chunk_dw_u900_ext.json.zst
      --split diag12\\submission_split.json --fused diag10\\cache_rrf3d2.json
      --ref-c120in diag14\\submission_c120in.json --pool cache_pool_rr.json
      --ref-cx110cov diag16\\submission_cx110cov.json --dq docwin_ext\\cache_docwin_ext.json
      --outdir diag17
  python scripts/cx130.py check-zip --json diag17\\submission_cx130cov.json
      --zip diag17\\submission_cx130cov.zip --report diag17\\report.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from cx110 import choose_delta, hist_from_pairs, kprime, norm_pairs, suggest_next_delta, total_chunks
from cx93 import B_COV, BAND_NAMES, WD, apply_bonus, filter_base, load_inputs
from diag14 import order_q, strip_meta
from diag14 import verify_zip as verify_zip_bytes
from dwx_cands import band_of

TARGET_MEAN = 130.0
MAX_ZIP = 93_000_000
DELTA_REF = 23  # delta cua cx110cov
CX110_JSON_BYTES = 505_693_628
CX110_ZIP_BYTES = 79_380_644
NAME = "cx130cov"


def uoc_zip(json_bytes: float, json_ref: int = CX110_JSON_BYTES, zip_ref: int = CX110_ZIP_BYTES) -> float:
    """Uoc kich thuoc zip LZMA theo ti le byte JSON/zip cua cx110cov (505.693.628 / 79.380.644)."""
    return json_bytes * zip_ref / json_ref


def largest_delta_within(hist: list, json_now: float, n_now: int, delta_hi: int, max_zip: int,
                         json_ref: int = CX110_JSON_BYTES, zip_ref: int = CX110_ZIP_BYTES) -> int:
    """delta <= delta_hi lon nhat voi zip uoc = json_now * N(delta)/N_now * (zip_ref/json_ref) <= max_zip.

    N(delta) = tong chunk theo hist [[k_q, |C_q|], so_cau]. Neu tat ca deu vuot thi tra 0.
    """
    if delta_hi <= 0 or n_now <= 0 or json_now <= 0:
        return 0
    lo, hi = 0, int(delta_hi)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        est = uoc_zip(json_now * total_chunks(hist, mid) / n_now, json_ref, zip_ref)
        if est <= max_zip:
            lo = mid
        else:
            hi = mid - 1
    return lo


def check_repro23(prep: dict, ref110: dict) -> dict:
    """Dung delta = 23 (k' = min(k_q+23, |C_q|)) tren ung vien ext, so Y NGUYEN voi submission_cx110cov.

    Tra {cau, cau_khop, ty_le_cau_khop, vi_tri_khop, ty_le_vi_tri_chunk_khop, ok, vi_du_lech}.
    """
    n = n_ok = n_pos = n_pos_tot = 0
    vi_du = []
    for q, want in ref110.items():
        p = prep.get(q)
        got = strip_meta(p["ob"], kprime(p["kq"], p["cap"], DELTA_REF)) if p else []
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
            vi_du.append({"id": q, "k_110": len(w), "got_n": len(g), "vi_tri_lech": i,
                          "got": g[i] if i is not None and i < len(g) else None,
                          "want": w[i] if i is not None and i < len(w) else None})
    return {"cau": n, "cau_khop": n_ok, "ty_le_cau_khop": n_ok / max(n, 1),
            "vi_tri_khop": n_pos, "ty_le_vi_tri_chunk_khop": n_pos / max(n_pos_tot, 1),
            "ok": n_ok == n and n == len(prep), "vi_du_lech": vi_du}


def build_sub(split: list, prep: dict, ref110: dict, dq: dict, rank: dict, delta: int,
              base: dict) -> tuple[list, dict]:
    """Dung submission_cx130cov: k'_q = min(k_q + delta, |C_q|) chunk dau cua order da thuong b.

    Tien to cx110cov: k110_q = min(k_q + 23, |C_q|) chunk dau phai trung submission_cx110cov (kiem
    ref110). Chunk them = vi tri k110_q..k'_q cua cung thu tu chon - phan tich theo hang doc trong
    split, logit, s, nguon (doc D_q cache_docwin_ext = ung vien moi docwin_ext; con lai = cu).
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
        k110 = kprime(kq, cap, DELTA_REF)
        sel = strip_meta(ob, kp)
        extra = ob[k110:kp]
        w = norm_pairs(ref110.get(q, []))
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
          "cau_tien_to_cx110cov_ok": n_prefix_ok,
          "tien_to_kiem_tra": delta >= DELTA_REF,
          "TB_chunk_cau": sum(kps) / n, "min_chunk": min(kps, default=0), "max_chunk": max(kps, default=0),
          "tong_chunk": sum(kps),
          "TB_doc_khac_nhau_co_chunk_cau": sum(n_doc_q) / n,
          "TB_doc_cau": sum(len(it["relevant_docs"]) for it in sub) / n,
          "TB_doc_them_cau": 0.0,
          "TB_chunk_them_cau": sum(n_them_q) / n,
          "chunk_them": {
              "vs": f"cx110cov (k110 = min(k_q + {DELTA_REF}, |C_q|))",
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


def _write_sub(sub: list, fp: Path) -> int:
    with fp.open("w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False)
    return fp.stat().st_size


def run(cands: str, split_path: str, fused_path: str, ref_path: str, pool_path: str,
        ref_cx110cov_path: str, dq_path: str, outdir: str, target_mean: float = TARGET_MEAN,
        max_zip: int = MAX_ZIP, delta: int | None = None, wd: float = WD, b_cov: float = B_COV,
        report_path: str = "") -> dict:
    t0 = time.time()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rr, split, fu, ref_chunks, base, rank, _top = load_inputs(cands, split_path, fused_path, ref_path, pool_path)
    ref110 = {str(it["id"]): it["relevant_chunks"]
              for it in json.loads(Path(ref_cx110cov_path).read_text(encoding="utf-8"))}
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
    repro = check_repro23(prep, ref110)
    print(f"tai lap delta=23: {json.dumps(repro, ensure_ascii=False)}", flush=True)
    if not repro["ok"]:
        rp.write_text(json.dumps({"quy_uoc": "cx130", "tai_lap_delta23": repro}, ensure_ascii=False),
                      encoding="utf-8")
        print(f"KHONG tai lap duoc cx110cov voi delta=23, dung lai; ghi {rp}", flush=True)
        raise SystemExit(2)
    hist = hist_from_pairs(pairs)
    n_ref = total_chunks(hist, DELTA_REF)
    delta_target, dat = choose_delta(pairs, target_mean)
    est_target = uoc_zip(CX110_JSON_BYTES * total_chunks(hist, delta_target) / max(n_ref, 1))
    if delta is not None:
        d_init, ghi_nhan = int(delta), True
        ly_do = f"ghi nhan --delta {d_init} (giam theo ngan sach zip {max_zip})"
    elif dat and est_target <= max_zip:
        d_init, ghi_nhan = delta_target, False
        ly_do = (f"delta nho nhat de TB k' >= {target_mean}: {delta_target}; "
                 f"uoc zip {est_target:.0f} <= {max_zip}")
    else:
        d_init = largest_delta_within(hist, CX110_JSON_BYTES, n_ref, delta_target, max_zip)
        ghi_nhan = False
        ly_do = ((f"khong dat duoc TB k' >= {target_mean} vi |C_q| chan (TB |C_q| = "
                  f"{sum(cap for _k, cap in pairs) / max(len(pairs), 1):.1f})")
                 if not dat else
                 (f"delta {delta_target} dat target nhung uoc zip {est_target:.0f} > {max_zip}"))
        ly_do += f"; lay delta lon nhat theo uoc tinh zip <= {max_zip}: {d_init}"
    print(f"chon delta: target {delta_target} (dat {dat}), uoc zip target {est_target:.0f} -> dung {d_init}",
          flush=True)
    fp = out / f"submission_{NAME}.json"
    tried = {}
    d_use = d_init
    for _lan in range(6):
        sub, st = build_sub(split, prep, ref110, dq, rank, d_use, base)
        jb = _write_sub(sub, fp)
        del sub
        est = uoc_zip(jb)
        tried[d_use] = est
        fits = est <= max_zip
        print(f"lan build {_lan + 1}: delta {d_use}, json {jb}, uoc zip {est:.0f}, duoi {max_zip}: {fits}",
              flush=True)
        if ghi_nhan:
            break
        if fits:
            cand = largest_delta_within(hist, jb, st["kprime"]["tong"], delta_target, max_zip)
            nxt = next((x for x in range(cand, d_use, -1) if x not in tried), None)
        else:
            if d_use <= 0:
                nxt = None
            else:
                cand = largest_delta_within(hist, jb, st["kprime"]["tong"], d_use - 1, max_zip)
                nxt = next((x for x in range(min(cand, d_use - 1), -1, -1) if x not in tried), None)
        if nxt is None:
            break
        d_use = nxt
    best_d = max((x for x, e in tried.items() if e <= max_zip), default=None)
    if best_d is None:
        best_d = min(tried)
    if best_d != d_use:
        d_use = best_d
        sub, st = build_sub(split, prep, ref110, dq, rank, d_use, base)
        jb = _write_sub(sub, fp)
        del sub
        tried[d_use] = uoc_zip(jb)
    est = tried[d_use]
    if d_use != d_init:
        ly_do += f"; uoc lai bang JSON that -> dung {d_use} (uoc zip {est:.0f})"
    uoc = [{"delta": x, "uoc_zip": e, "duoi_gioi_han": e <= max_zip} for x, e in sorted(tried.items())]
    st["json_bytes"] = fp.stat().st_size
    report = {"quy_uoc": {
                  "giong": "cx110cov: doc nen = relevant_docs split (0 doc them), ung vien "
                           "chunk_dw_u900_ext.json.zst, s = zc + 4*zd (diag14.order_q), b = 1,0 (apply_bonus)",
                  "khac": "k'_q = min(k_q + delta, |C_q|) voi delta chon lai cho TB k' >= 130,0",
                  "k_q": "so relevant_chunks cua cau trong submission_c120in.json",
                  "tien_to": "k110 = min(k_q + 23, |C_q|) chunk dau phai trung submission_cx110cov",
                  "chunk_them": f"vi tri sau {DELTA_REF} chunk tien to cx110cov cua cung thu tu chon",
                  "nguon_moi": "doc D_q cua cache_docwin_ext.json = ung vien moi docwin_ext; con lai = cu",
                  "uoc_zip": "ti le byte JSON/zip cua cx110cov (505.693.628 / 79.380.644)",
                  "zip": "zip_submission.py --method lzma, entry submission.json"},
              "wd": wd, "b_cov": b_cov, "target_mean": target_mean, "max_zip": max_zip,
              "delta_ref_cx110cov": DELTA_REF,
              "cau": len(split), "k_q_TB": sum(k for k, _ in pairs) / max(len(pairs), 1),
              "tai_lap_delta23": repro,
              "delta": {"target": delta_target, "dung": d_use, "dat_target": dat,
                        "ly_do": ly_do, "uoc_zip_target": est_target, "ghi_nhan": ghi_nhan,
                        "thu": thu},
              "kprime": st["kprime"], "cau_chan_boi_C_q": st["cau_chan_boi_C_q"],
              NAME: st,
              "uoc_zip": {"json_bytes": st["json_bytes"], "est_zip": est,
                          "duoi_gioi_han": est <= max_zip,
                          "ty_le_json_zip": CX110_JSON_BYTES / CX110_ZIP_BYTES, "cac_lan": uoc},
              "hist_kcap": hist_from_pairs(pairs),
              "thoi_gian_giay": round(time.time() - t0, 1)}
    rp.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"delta {delta_target} -> dung {d_use}: {json.dumps(st, ensure_ascii=False)}", flush=True)
    print(f"ghi {rp}; tong {report['thoi_gian_giay']}s", flush=True)
    if st["tien_to_kiem_tra"] and st["cau_tien_to_cx110cov_ok"] != len(split):
        print(f"TIEN TO cx110cov sai {st['cau_tien_to_cx110cov_ok']}/{len(split)}, dung lai", flush=True)
        raise SystemExit(2)
    if est > max_zip:
        if d_use > 0:
            nxt = largest_delta_within(hist, st["json_bytes"], st["kprime"]["tong"], d_use - 1, max_zip)
            (fp.parent / "next_delta.txt").write_text(f"--delta {nxt}", encoding="utf-8")
            print(f"uoc zip {est:.0f} > {max_zip}; ghi next_delta.txt --delta {nxt}", flush=True)
        else:
            print(f"uoc zip {est:.0f} > {max_zip} nhung delta da 0, khong giam duoc", flush=True)
    return report


def check_zip(json_path: str, zip_path: str, report_path: str, max_zip: int = MAX_ZIP) -> int:
    """Doc lai zip == JSON goc + zip <= max_zip; cap nhat report. Tra exit code (0/1/3/4)."""
    p, zp = Path(json_path), Path(zip_path)
    st = dict(verify_zip_bytes(str(p), str(zp)))
    zb, jb = zp.stat().st_size, p.stat().st_size
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    duoi = zb <= max_zip
    ok_noi_dung = bool(st["ok"])
    rp = Path(report_path)
    report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    d_used = report.get("delta", {}).get("dung")
    n_chunks = report.get(NAME, {}).get("tong_chunk", 0)
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
            next_d = suggest_next_delta(report.get("hist_kcap", []), int(d_used), n_chunks, zb, max_zip, marg)
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
    rn.add_argument("--ref-cx110cov", required=True, help="submission_cx110cov.json (kiem tai lap delta=23)")
    rn.add_argument("--dq", required=True, help="cache_docwin_ext.json (D_q = ung vien moi)")
    rn.add_argument("--outdir", required=True)
    rn.add_argument("--report", default="", help="JSON bao cao (mac dinh <outdir>/report.json)")
    rn.add_argument("--target-mean", type=float, default=TARGET_MEAN)
    rn.add_argument("--max-zip", type=int, default=MAX_ZIP, help="gioi han zip LZMA (mac dinh 93.000.000)")
    rn.add_argument("--delta", type=int, default=None, help="ghi nhan delta (mac dinh tinh tu target)")
    rn.add_argument("--wd", type=float, default=WD)
    rn.add_argument("--b-cov", type=float, default=B_COV)
    cz = sp.add_parser("check-zip")
    cz.add_argument("--json", required=True)
    cz.add_argument("--zip", required=True)
    cz.add_argument("--report", required=True)
    cz.add_argument("--max-zip", type=int, default=MAX_ZIP)
    a = ap.parse_args()
    if a.cmd == "run":
        rep = run(a.cands, a.split, a.fused, a.ref_c120in, a.pool, a.ref_cx110cov, a.dq, a.outdir,
                  a.target_mean, a.max_zip, a.delta, a.wd, a.b_cov, a.report)
        if rep["uoc_zip"]["duoi_gioi_han"]:
            raise SystemExit(0)
        raise SystemExit(4 if rep["delta"]["dung"] <= 0 else 3)
    raise SystemExit(check_zip(a.json, a.zip, a.report, a.max_zip))


if __name__ == "__main__":
    main()
