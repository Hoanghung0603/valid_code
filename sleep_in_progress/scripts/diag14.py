"""diag14: dựng 3 bài nộp nhiều chunk hơn từ ứng viên U900 (CPU trên b660), kiểm tái lập trước.

run  kiểm tái lập với k_q gốc (phải trùng y nguyên chunks_rrf3d2_cf4.json; không 100% thì dừng),
     rồi dựng 3 bài cùng doc nền như submission_split (relevant_docs của split + doc nguồn chunk
     còn thiếu nối cuối; c120in giữ nguyên doc của split):
       c120   toàn bộ chunk ứng viên của câu (<=120, bỏ trùng nguyên văn), sắp s = zc + 4*zd
       c120in như c120 nhưng chỉ giữ chunk có doc_id trong relevant_docs của split (không thêm doc)
       ck105  min(k_q + 30, số ứng viên) chunk đầu theo s, thêm doc nguồn như c120
     + phân tích chunk thêm theo dải hạng-s (76-90, 91-105, 106-120). --cap K giảm k đều mọi câu.
verify-zip  đọc lại zip (zipfile + json.load) so với JSON gốc.

Dùng:
  python scripts/diag14.py run --split submission_split.json --rerank chunk_dw_u900.json.zst \
      --fused cache_rrf3d2.json --ref-chunks chunks_d900t015.json --target chunks_rrf3d2_cf4.json \
      --outdir D:\\r2ai\\data\\sub\\diag14
  python scripts/diag14.py verify-zip --json submission_c120.json --zip submission_c120.zip
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

from chunk_docfuse import doc_z, pick_q, zscores
from chunks_from_rerank import load_rerank
from diag13 import with_chunk_docs

BANDS = ((76, 90, "76-90"), (91, 105, "91-105"), (106, 120, "106-120"))
MODES = ("c120", "c120in", "ck105")
ZIP_LIMIT = 95_000_000


def order_q(rows: list, fused_hits: list, wd: float = 4.0) -> list[dict]:
    """Toàn bộ chunk ứng viên: s = zc + wd*zd, sắp s giảm dần (đồng điểm giữ thứ tự cũ), bỏ trùng
    nguyên văn. Mỗi phần tử: doc_id, chunk_text, logit, zc, zd, s. Cùng thứ tự với pick_q."""
    zc = zscores([float(r[3]) for r in rows])
    zdm = doc_z(fused_hits)
    zmin = min(zdm.values()) if zdm else 0.0
    scored = [(c + wd * zdm.get(str(r[0]), zmin), c, zdm.get(str(r[0]), zmin), r) for c, r in zip(zc, rows)]
    order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
    out, seen = [], set()
    for i in order:
        s, c, z, r = scored[i]
        if r[4] in seen:
            continue
        seen.add(r[4])
        out.append({"doc_id": r[0], "chunk_text": r[4], "logit": float(r[3]), "zc": c, "zd": z, "s": s})
    return out


def strip_meta(order: list[dict], k: int | None = None) -> list[dict]:
    sel = order if k is None else order[:k]
    return [{"doc_id": c["doc_id"], "chunk_text": c["chunk_text"]} for c in sel]


def repro_check(rr: dict, fu: dict, ref: list, target: list, wd: float = 4.0) -> dict:
    """So nguyên văn chunk với k_q gốc giữa (a) pick_q gốc, (b) order_q[:k] với target. Tỉ lệ khớp."""
    tm = {str(it["id"]): it["relevant_chunks"] for it in target}
    n_q = n_ok_pick = n_ok_ord = n_pos = n_pos_tot = 0
    vi_du = []
    for it in ref:
        q = str(it["id"])
        k = len(it["relevant_chunks"])
        want = tm.get(q)
        rows, hits = rr.get(q, []), fu.get(q, [])
        got_pick = pick_q(rows, hits, k, wd)
        got_ord = strip_meta(order_q(rows, hits, wd), k)
        n_q += 1
        ok_p, ok_o = got_pick == want, got_ord == want
        n_ok_pick += ok_p
        n_ok_ord += ok_o
        for x, y in zip(got_ord, want or []):
            n_pos_tot += 1
            n_pos += x == y
        n_pos_tot += abs(len(got_ord) - len(want or []))
        if not (ok_p and ok_o) and len(vi_du) < 5:
            vi_du.append({"id": it["id"], "k": k, "want_n": len(want or []), "got_n": len(got_ord),
                          "khop_pick": ok_p, "khop_order": ok_o})
    return {"cau": n_q, "cau_khop_pick_q": n_ok_pick, "cau_khop_order_q": n_ok_ord,
            "ty_le_cau_khop": n_ok_ord / max(n_q, 1), "ty_le_vi_tri_chunk_khop": n_pos / max(n_pos_tot, 1),
            "ok": n_ok_pick == n_q and n_ok_ord == n_q, "vi_du_lech": vi_du}


def build_sub(split: list, orders: dict, kq: dict, mode: str, delta: int = 30,
              cap: int = 120) -> tuple[list, dict]:
    """Dựng 1 bài. mode c120/ck105: doc = doc split + doc nguồn chunk còn thiếu; c120in: doc giữ nguyên."""
    if mode not in MODES:
        raise ValueError(f"mode la: {mode}")
    out, added, nch, ndc = [], [], [], []
    for it in split:
        q = str(it["id"])
        order = (orders.get(q) or [])[:cap]
        if mode == "c120":
            chunks = strip_meta(order)
            docs, add = with_chunk_docs(it["relevant_docs"], chunks)
        elif mode == "ck105":
            k = min(kq.get(q, 0) + delta, len(order))
            chunks = strip_meta(order, k)
            docs, add = with_chunk_docs(it["relevant_docs"], chunks)
        else:
            allowed = {str(d) for d in it["relevant_docs"]}
            chunks = [c for c in strip_meta(order) if str(c["doc_id"]) in allowed]
            docs, add = list(it["relevant_docs"]), 0
        out.append({"id": it["id"], "relevant_docs": docs, "relevant_chunks": chunks})
        added.append(add)
        nch.append(len(chunks))
        ndc.append(len({str(c["doc_id"]) for c in chunks}))
    n = max(len(out), 1)
    st = {"mode": mode, "cau": len(out),
          "TB_chunk_cau": sum(nch) / n, "min_chunk": min(nch) if nch else 0, "max_chunk": max(nch) if nch else 0,
          "TB_doc_cau": sum(len(s["relevant_docs"]) for s in out) / n,
          "TB_doc_them_cau": sum(added) / n, "tong_doc_them": sum(added),
          "TB_doc_khac_nhau_co_chunk_cau": sum(ndc) / n}
    return out, st


def analyze_bands(orders: dict, kq: dict, split_docs: dict) -> dict:
    """Chunk theo dải hạng-s: TB logit, TB zd, tỉ lệ doc đã có trong danh sách nộp (doc của split)."""
    acc = {name: {"n": 0, "logit": 0.0, "zd": 0.0, "in_doc": 0, "them": 0} for _, _, name in BANDS}
    tong_them = 0
    for q, order in orders.items():
        allowed = split_docs.get(q, set())
        k = kq.get(q, 0)
        for rank, c in enumerate(order, 1):
            if rank > k:
                tong_them += 1
            for a, b, name in BANDS:
                if a <= rank <= b:
                    e = acc[name]
                    e["n"] += 1
                    e["logit"] += c["logit"]
                    e["zd"] += c["zd"]
                    e["in_doc"] += str(c["doc_id"]) in allowed
                    e["them"] += rank > k
                    break
    n = max(len(orders), 1)
    out = {}
    for _, _, name in BANDS:
        e = acc[name]
        out[name] = {"so_chunk": e["n"], "TB_chunk_moi_cau": e["n"] / n,
                     "TB_logit": e["logit"] / e["n"] if e["n"] else None,
                     "TB_zd": e["zd"] / e["n"] if e["n"] else None,
                     "ti_le_doc_trong_danh_sach_nop": e["in_doc"] / e["n"] if e["n"] else None,
                     "ti_le_thuc_su_them_hon_k_q": e["them"] / e["n"] if e["n"] else None}
    out["tong"] = {"chunk_hang_76_120": sum(v["so_chunk"] for k2, v in out.items() if k2 != "tong"),
                   "chunk_them_hon_k_q_toan_bo": tong_them, "TB_chunk_them_hon_k_q_moi_cau": tong_them / n}
    return out


def verify_zip(json_path, zip_path) -> dict:
    """Đọc lại zip bằng zipfile + json.load, so với JSON gốc."""
    p, zp = Path(json_path), Path(zip_path)
    raw = p.read_bytes()
    parsed = json.loads(raw)
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        data = z.read("submission.json")
    bang_bytes = data == raw
    obj = json.loads(data)
    bang_json = obj == parsed
    del raw, parsed, data, obj
    ok = names == ["submission.json"] and bang_bytes and bang_json
    return {"entry": names, "bang_goc_bytes": bang_bytes, "bang_goc_json": bang_json, "ok": ok}


def run(split_path: str, rerank_path: str, fused_path: str, ref_path: str, target_path: str,
        outdir: str, wd: float = 4.0, delta: int = 30, cap: int = 120) -> tuple[dict, int]:
    """Kiểm tái lập rồi dựng 3 bài vào outdir. Trả (report, exit_code: 0 ok, 2 lệch tái lập)."""
    t0 = time.time()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rerank = load_rerank(rerank_path)
    fused = json.loads(Path(fused_path).read_text(encoding="utf-8"))
    ref = json.loads(Path(ref_path).read_text(encoding="utf-8"))
    target = json.loads(Path(target_path).read_text(encoding="utf-8"))
    split = json.loads(Path(split_path).read_text(encoding="utf-8"))
    rr = {str(q): v for q, v in rerank.items()}
    fu = {str(q): v for q, v in fused.items()}
    del rerank, fused
    kq = {str(it["id"]): len(it["relevant_chunks"]) for it in ref}
    print(f"nap xong {time.time() - t0:.0f}s: {len(split)} cau, k_q TB {sum(kq.values()) / max(len(kq), 1):.1f}",
          flush=True)
    rep = repro_check(rr, fu, ref, target, wd)
    print(f"tai lap: {json.dumps(rep, ensure_ascii=False)}", flush=True)
    if not rep["ok"]:
        return {"repro": rep}, 2
    del target, ref
    orders = {str(it["id"]): order_q(rr.get(str(it["id"]), []), fu.get(str(it["id"]), []), wd) for it in split}
    split_docs = {str(it["id"]): {str(d) for d in it["relevant_docs"]} for it in split}
    report = {"repro": rep, "wd": wd, "delta_ck105": delta, "cap_k": cap,
              "zip_limit": ZIP_LIMIT, "bai": {}}
    for mode in MODES:
        sub, st = build_sub(split, orders, kq, mode, delta, cap)
        fp = out / f"submission_{mode}.json"
        with fp.open("w", encoding="utf-8") as f:
            json.dump(sub, f, ensure_ascii=False)
        st["json_bytes"] = fp.stat().st_size
        report["bai"][mode] = st
        print(f"{mode}: {json.dumps(st, ensure_ascii=False)}", flush=True)
        del sub
    report["phan_tich_hang_s"] = analyze_bands(orders, kq, split_docs)
    report["thoi_gian_giay"] = round(time.time() - t0, 1)
    print(f"phan tich: {json.dumps(report['phan_tich_hang_s'], ensure_ascii=False)}", flush=True)
    print(f"xong {report['thoi_gian_giay']}s", flush=True)
    return report, 0


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    rn = sp.add_parser("run")
    rn.add_argument("--split", required=True, help="submission_split.json (doc nền)")
    rn.add_argument("--rerank", required=True, help="chunk_dw_u900.json.zst")
    rn.add_argument("--fused", required=True, help="cache_rrf3d2.json (nguồn zd)")
    rn.add_argument("--ref-chunks", required=True, help="chunks_d900t015.json (k_q)")
    rn.add_argument("--target", required=True, help="chunks_rrf3d2_cf4.json (kiểm tái lập)")
    rn.add_argument("--outdir", required=True)
    rn.add_argument("--report", default="")
    rn.add_argument("--wd", type=float, default=4.0)
    rn.add_argument("--delta", type=int, default=30, help="ck105 = min(k_q + delta, số ứng viên)")
    rn.add_argument("--cap", type=int, default=120, help="giảm k đều mọi câu nếu zip vượt 95MB")
    vz = sp.add_parser("verify-zip")
    vz.add_argument("--json", required=True)
    vz.add_argument("--zip", required=True)
    a = ap.parse_args()
    if a.cmd == "run":
        rep, code = run(a.split, a.rerank, a.fused, a.ref_chunks, a.target, a.outdir, a.wd, a.delta, a.cap)
        rp = a.report or str(Path(a.outdir) / "report.json")
        Path(rp).write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
        print(f"ghi {rp}", flush=True)
        raise SystemExit(code)
    st = verify_zip(a.json, a.zip)
    print(json.dumps(st, ensure_ascii=False), flush=True)
    raise SystemExit(0 if st["ok"] else 1)


if __name__ == "__main__":
    main()
