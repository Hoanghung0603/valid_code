"""cxp3: bài nộp pool3 (diag18) — Phần A đổi danh sách doc, Phần B chọn chunk lại trên ứng viên mở rộng.

Phần A:
  mknq     dựng doclist_p3_nq: mỗi câu lấy top-n_q doc ĐẦU theo thứ hạng --rank (cache_fhyde3),
           n_q = đúng số doc của câu đó trong submission_split. In TB số doc NEW lọt/câu
           (doc NEW nằm trong top-n_q) và TB số doc bị đẩy ra/câu (doc của split không còn
           trong top-n_q).
  swapdoc  submission_p3t20_c130 / p3t16_c130 / p3nq_c130 — relevant_docs = doclist_p3
           (+ doc nguồn chunk của cx130cov còn thiếu, nối cuối), relevant_chunks GIỮ Y HỆT
           submission_cx130cov. Truyền --new để in TB NEW/câu (doc NEW trong relevant_docs).

Phần B (run): cùng công thức cx130cov trên ứng viên chunk_dw_u900_p3 với doc nền = doclist:
  s = zc + 4*zd; zc = z-score logit trên toàn bộ ứng viên của câu (quy ước diag14.order_q);
  zd = z-score điểm hợp nhất: doc có trong --fused (cache_rrf3d2, như cx130cov) -> z của cache đó;
      doc không có trong --fused nhưng có trong --fused-new (mặc định cache_rrf3d2_p3 = rrf3d2
      TÁI LẬP trên pool3: fuse3 bge + Qwen8B + dense*2, k=60 — 8B vắng với doc NEW nên NEW chỉ
      được bge+dense đóng góp) -> z của cache mới; còn lại zd = zmin của câu (như diag14).
      Ghi rõ cách trong report.quy_uoc.zd.
  b = 1,0 thưởng chunk đầu mỗi doc (cx93.apply_bonus); k'_q = min(k_q + delta, |C_q|) với
  k_q = số relevant_chunks của submission_c120in.json; |C_q| = ứng viên sau lọc doc nền + bỏ trùng
  nguyên văn. delta bắt đầu 57 (cx130cov), chỉ giảm khi uoc zip LZMA > --max-zip (93.000.000).

repro  chạy đúng công thức trên doc/ứng viên CŨ (relevant_docs của cx130cov + chunk_dw_u900_ext,
  zd rrf3d2, delta 57) — phải tái lập relevant_chunks của submission_cx130cov >= --min-match số câu.

check-zip  đọc lại zip (zipfile + json.load) == JSON gốc, zip <= max_zip, cập nhật report; exit 0/1/3/4
  (3 = vượt -> ghi next_delta_<name>.txt; 4 = vượt nhưng delta đã 0).

Dùng:
  python scripts/cxp3.py mknq --rank diag18\\cache_fhyde3.json --split diag12\\submission_split.json \
      --new diag18\\new_cands.json --out diag18\\doclist_p3_nq.json --report diag18\\doclist_p3_nq_report.json
  python scripts/cxp3.py swapdoc --cx130 diag17\\submission_cx130cov.json --new diag18\\new_cands.json \
      --doclist submission_p3t20_c130=diag18\\doclist_p3_t20.json \
      --doclist submission_p3t16_c130=diag18\\doclist_p3_t16.json \
      --doclist submission_p3nq_c130=diag18\\doclist_p3_nq.json --outdir diag18
  python scripts/cxp3.py repro --cands docwin_ext\\chunk_dw_u900_ext.json.zst
      --ref-cx130cov diag17\\submission_cx130cov.json --fused diag10\\cache_rrf3d2.json
      --ref-c120in diag14\\submission_c120in.json --report diag18\\repro_cx130.json
  python scripts/cxp3.py run --cands diag18\\chunk_dw_u900_p3.json.zst
      --doclist submission_p3t20_cx=diag18\\doclist_p3_t20.json --fused diag10\\cache_rrf3d2.json
      --fused-new diag18\\cache_fhyde3.json --ref-c120in diag14\\submission_c120in.json
      --new-cands diag18\\new_cands.json --outdir diag18
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from chunk_docfuse import doc_z, zscores
from chunks_from_rerank import load_rerank
from cx110 import hist_from_pairs, kprime, norm_pairs
from cx130 import largest_delta_within, uoc_zip
from cx93 import B_COV, WD, apply_bonus, filter_base
from diag14 import strip_meta
from diag14 import verify_zip as verify_zip_bytes

DELTA_REF = 57  # delta cua cx130cov
CX110_JSON_BYTES = 505_693_628
CX110_ZIP_BYTES = 79_380_644
MAX_ZIP = 93_000_000


def order_q2(rows: list, fused_hits: list, fused_new_hits: list | None = None, wd: float = WD) -> list[dict]:
    """như diag14.order_q nhưng zd hỗn hợp 2 cache (fused_new_hits rỗng/None -> y order_q).

    zd: doc có trong fused_hits -> z của fused_hits; doc chỉ có trong fused_new_hits -> z cache mới;
    còn lại zmin = min z của fused_hits.
    """
    zc = zscores([float(r[3]) for r in rows])
    zdm = doc_z(fused_hits)
    zmin = min(zdm.values()) if zdm else 0.0
    znew = doc_z(fused_new_hits) if fused_new_hits else {}

    def zof(d) -> float:
        s = str(d)
        if s in zdm:
            return zdm[s]
        if s in znew:
            return znew[s]
        return zmin

    scored = [(c + wd * zof(r[0]), c, zof(r[0]), r) for c, r in zip(zc, rows)]
    order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
    out, seen = [], set()
    for i in order:
        s, c, z, r = scored[i]
        if r[4] in seen:
            continue
        seen.add(r[4])
        out.append({"doc_id": r[0], "chunk_text": r[4], "logit": float(r[3]), "zc": c, "zd": z, "s": s})
    return out


def parse_doclists(pairs: list) -> dict:
    """['NAME=path', ...] -> {NAME: path} (NAME = tiền tố file submission_NAME.json)."""
    out = {}
    for item in pairs:
        name, _, path = item.partition("=")
        if not name or not path:
            raise ValueError(f"--doclist cần dạng NAME=path, nhận: {item}")
        out[name] = path
    return out


def load_json(path: str):
    p = Path(path)
    if p.name.endswith(".zst"):
        import zstandard
        with open(p, "rb") as f:
            return json.loads(zstandard.ZstdDecompressor().stream_reader(f).read())
    return json.loads(p.read_text(encoding="utf-8"))


def swapdoc(cx130_path: str, doclists: dict, outdir: str, new_path: str = "") -> dict:
    """Phần A: đổi relevant_docs theo doclist, GIỮ nguyên relevant_chunks của cx130cov.

    Doc nguồn chunk không nằm trong doclist được nối cuối relevant_docs (thứ tự xuất hiện chunk).
    new_path (new_cands.json): đếm doc NEW trong relevant_docs cuối (TB NEW/câu).
    Trả report {name: thống kê}.
    """
    t0 = time.time()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    ref = load_json(cx130_path)
    new_ids = ({str(q): {str(d) for d in v} for q, v in load_json(new_path).items()} if new_path else {})
    report = {"quy_uoc": {"doc": "doclist_p3 + doc nguồn chunk cx130cov còn thiếu (nối cuối)",
                          "chunk": "giữ y hệt submission_cx130cov", "cx130": cx130_path,
                          "new": new_path or "(không đếm NEW)"},
              "bai": {}}
    for name, path in doclists.items():
        dl = {str(k): v for k, v in load_json(path).items()}
        sub, n_noi, doc_cau, n_new = [], [], [], []
        for it in ref:
            q = str(it["id"])
            docs = list(dl[q])
            seen = {str(d) for d in docs}
            noi = []
            for c in it["relevant_chunks"]:
                d = c["doc_id"]
                if str(d) not in seen:
                    seen.add(str(d))
                    noi.append(d)
            assert all(str(c["doc_id"]) in seen for c in it["relevant_chunks"])
            sub.append({"id": it["id"], "relevant_docs": docs + noi,
                        "relevant_chunks": it["relevant_chunks"]})
            n_noi.append(len(noi))
            doc_cau.append(len(docs) + len(noi))
            nset = new_ids.get(q, set())
            n_new.append(sum(1 for d in docs + noi if str(d) in nset))
        fp = out / f"{name}.json"
        with fp.open("w", encoding="utf-8") as f:
            json.dump(sub, f, ensure_ascii=False)
        n = max(len(sub), 1)
        st = {"cau": len(sub), "doclist": path,
              "TB_doc_cau": sum(doc_cau) / n, "min_doc": min(doc_cau), "max_doc": max(doc_cau),
              "TB_NEW_cau": sum(n_new) / n if new_ids else None,
              "TB_doc_noi_cau": sum(n_noi) / n, "cau_co_doc_noi": sum(1 for x in n_noi if x),
              "max_doc_noi": max(n_noi, default=0), "tong_doc_noi": sum(n_noi),
              "chunk_giong_cx130cov": all(a["relevant_chunks"] == b["relevant_chunks"]
                                          for a, b in zip(sub, ref)),
              "json_bytes": fp.stat().st_size}
        assert st["chunk_giong_cx130cov"] and st["cau"] == len(ref)
        report["bai"][name] = st
        print(f"{name}: {json.dumps(st, ensure_ascii=False)}", flush=True)
    report["thoi_gian_giay"] = round(time.time() - t0, 1)
    (out / "p3a_report.json").write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"ghi {out / 'p3a_report.json'}; tong {report['thoi_gian_giay']}s", flush=True)
    return report


def mknq(rank_path: str, split_path: str, new_path: str, out_path: str, report_path: str) -> dict:
    """Phần A p3nq: doclist = top-n_q doc ĐẦU theo thứ hạng rank (cache_fhyde3), n_q = đúng số doc
    của câu đó trong submission_split.

    Thống kê: TB số doc NEW lọt/câu (doc NEW nằm trong top-n_q), TB số doc bị đẩy ra/câu
    (doc của split không còn trong top-n_q — bằng số doc mới chen vào vì n_q giữ nguyên).
    """
    t0 = time.time()
    rank = {str(k): v for k, v in load_json(rank_path).items()}
    split = load_json(split_path)
    new_ids = ({str(q): {str(d) for d in v} for q, v in load_json(new_path).items()} if new_path else {})
    out, ns, n_new, n_ra, n_q_tb = {}, [], [], [], []
    for it in split:
        q = str(it["id"])
        n_q = len(it["relevant_docs"])
        docs, seen = [], set()
        for h in rank.get(q, []):
            d = h[0]
            if d in seen:
                continue
            seen.add(d)
            docs.append(d)
            if len(docs) >= n_q:
                break
        out[q] = docs
        nset = new_ids.get(q, set())
        split_set = {str(d) for d in it["relevant_docs"]}
        seen_str = {str(d) for d in seen}
        ns.append(len(docs))
        n_q_tb.append(n_q)
        n_new.append(sum(1 for d in docs if str(d) in nset))
        n_ra.append(len(split_set - seen_str))
    n = max(len(out), 1)
    rep = {"cau": len(out), "rank": rank_path, "split": split_path,
           "n_q_TB": sum(n_q_tb) / n, "n_q_min": min(n_q_tb), "n_q_max": max(n_q_tb),
           "TB_doc_cau": sum(ns) / n, "min_doc": min(ns), "max_doc": max(ns),
           "TB_NEW_lot_cau": sum(n_new) / n, "tong_NEW_lot": sum(n_new),
           "TB_doc_day_ra_cau": sum(n_ra) / n, "tong_doc_day_ra": sum(n_ra),
           "cau_thieu_hang": sum(1 for x in ns if x < 0),
           "thoi_gian_giay": round(time.time() - t0, 1)}
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    Path(report_path).write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    print(f"ghi {out_path} + {report_path}", flush=True)
    return rep


def prep_queries(rr: dict, doclist: dict, fu: dict, fu_new: dict, kq: dict,
                 wd: float, b_cov: float) -> dict:
    """{qid: {kq, cap, ob}} — ob = order_q2 + thưởng b + lọc doc nền (như cx130.run)."""
    prep = {}
    for q, docs in doclist.items():
        base = {str(d) for d in docs}
        rows = rr.get(q, [])
        order = order_q2(rows, fu.get(q, []), fu_new.get(q, []), wd)
        cin = filter_base(order, base)
        prep[q] = {"kq": kq.get(q, 0), "cap": len(cin), "ob": apply_bonus(cin, b_cov)}
    return prep


def build_sub(qids: list, doclist: dict, prep: dict, new_ids: dict, delta: int) -> tuple[list, dict]:
    """Dựng submission: relevant_docs = doclist, k'_q chunk đầu của ob. Trả (sub, stats)."""
    sub = []
    kps, n_doc_q, n_new_q, n_them_q = [], [], [], []
    blocked = 0
    for q in qids:
        p = prep[q]
        docs = list(doclist[q])
        base = {str(d) for d in docs}
        kp = kprime(p["kq"], p["cap"], delta)
        sel = strip_meta(p["ob"], kp)
        for c in sel:
            assert str(c["doc_id"]) in base, f"chunk ngoai doclist: {q} {c['doc_id']}"
        nset = new_ids.get(q, set())
        kps.append(kp)
        n_doc_q.append(len({str(c["doc_id"]) for c in sel}))
        n_new_q.append(sum(1 for c in sel if str(c["doc_id"]) in nset))
        n_them_q.append(max(0, kp - p["kq"]))
        blocked += p["cap"] < p["kq"] + delta
        sub.append({"id": int(q) if str(q).isdigit() else q,
                    "relevant_docs": docs, "relevant_chunks": sel})
    n = max(len(sub), 1)
    st = {"kprime": {"TB": sum(kps) / n, "min": min(kps, default=0), "max": max(kps, default=0),
                     "tong": sum(kps)},
          "cau_chan_boi_C_q": blocked,
          "TB_chunk_cau": sum(kps) / n, "min_chunk": min(kps, default=0), "max_chunk": max(kps, default=0),
          "tong_chunk": sum(kps),
          "TB_doc_cau": sum(len(doclist[q]) for q in qids) / n,
          "TB_doc_co_chunk_cau": sum(n_doc_q) / n,
          "TB_chunk_tu_doc_NEW_cau": sum(n_new_q) / n,
          "TB_chunk_them_hon_k_q_cau": sum(n_them_q) / n}
    return sub, st


def _write_sub(sub: list, fp: Path) -> int:
    with fp.open("w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False)
    return fp.stat().st_size


def run(cands: str, doclists: dict, fused_path: str, fused_new_path: str, ref_path: str,
        outdir: str, new_cands_path: str = "", delta: int | None = None,
        delta_start: int = DELTA_REF, max_zip: int = MAX_ZIP, wd: float = WD, b_cov: float = B_COV,
        report_path: str = "") -> dict:
    """Phần B: dựng submission_<doclist> cho mỗi doclist. Trả report {name: {...}}."""
    t0 = time.time()
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rr = {str(q): v for q, v in load_rerank(cands).items()}
    fu = {str(q): v for q, v in load_json(fused_path).items()}
    fu_new = {str(q): v for q, v in load_json(fused_new_path).items()} if fused_new_path else {}
    ref = load_json(ref_path)
    kq = {str(it["id"]): len(it["relevant_chunks"]) for it in ref}
    new_ids = ({str(q): {str(d) for d in v} for q, v in load_json(new_cands_path).items()}
               if new_cands_path else {})
    dls = {name: {str(k): v for k, v in load_json(path).items()} for name, path in doclists.items()}
    print(f"nap xong {time.time() - t0:.0f}s: {len(rr)} cau ung vien, k_q TB "
          f"{sum(kq.values()) / max(len(kq), 1):.2f}", flush=True)
    report = {"quy_uoc": {
                  "giong": "cx130cov: s = zc + 4*zd (diag14.order_q), b = 1,0 (apply_bonus), "
                           "k'_q = min(k_q + delta, |C_q|), k_q = so relevant_chunks cua c120in",
                  "khac": "doc nen = doclist_p3_t{20,16}/p3nq (khong phai split); ung vien "
                          "chunk_dw_u900_p3",
                  "zd": "doc co trong cache_rrf3d2 -> z cua cache do (y cx130cov); doc khong co "
                        "trong rrf3d2 nhung co trong cache_rrf3d2_p3 (rrf3d2 TAI LAP tren pool3: "
                        "fuse3 bge + Qwen8B + dense*2, k=60; 8B vang voi doc NEW nen NEW chi duoc "
                        "bge+dense dong gop) -> z cua cache moi; con lai zd = zmin cua cau (diag14). "
                        "Z tinh rieng tren tung cache theo cau (chunk_docfuse.doc_z)",
                  "zip": "zip_submission.py --method lzma, entry submission.json"},
              "wd": wd, "b_cov": b_cov, "delta_start": delta_start, "max_zip": max_zip,
              "cands": cands, "fused": fused_path, "fused_new": fused_new_path,
              "cau": len(kq), "k_q_TB": sum(kq.values()) / max(len(kq), 1), "bai": {}}
    for name, doclist in dls.items():
        prep = prep_queries(rr, doclist, fu, fu_new, kq, wd, b_cov)
        pairs_hist = [(prep[q]["kq"], prep[q]["cap"]) for q in doclist]
        hist = hist_from_pairs(pairs_hist)
        fp = out / f"{name}.json"
        thu = []
        d_use = int(delta) if delta is not None else int(delta_start)
        ghi_nhan = delta is not None
        st = {}
        for _lan in range(6):
            sub, st = build_sub(sorted(doclist, key=lambda x: int(x) if str(x).isdigit() else x),
                                doclist, prep, new_ids, d_use)
            jb = _write_sub(sub, fp)
            del sub
            est = uoc_zip(jb, CX110_JSON_BYTES, CX110_ZIP_BYTES)
            thu.append({"delta": d_use, "json_bytes": jb, "uoc_zip": est,
                        "duoi_gioi_han": est <= max_zip})
            print(f"{name} lan build {_lan + 1}: delta {d_use}, json {jb}, uoc zip {est:.0f}, "
                  f"duoi {max_zip}: {est <= max_zip}", flush=True)
            if ghi_nhan or est <= max_zip or d_use <= 0:
                break
            d_use = largest_delta_within(hist, jb, st["tong_chunk"], d_use - 1, max_zip,
                                         CX110_JSON_BYTES, CX110_ZIP_BYTES)
        st["json_bytes"] = fp.stat().st_size
        st["hist_kcap"] = hist
        report["bai"][name] = {
            **st, "doclist": doclists[name], "delta": {"dung": d_use, "ghi_nhan": ghi_nhan,
                                                       "thu": thu},
            "uoc_zip": {"est_zip": thu[-1]["uoc_zip"], "duoi_gioi_han": thu[-1]["duoi_gioi_han"],
                        "ty_le_json_zip": CX110_JSON_BYTES / CX110_ZIP_BYTES}}
        print(f"{name}: delta {d_use}: {json.dumps(st, ensure_ascii=False)}", flush=True)
        del prep
    del rr, fu
    report["thoi_gian_giay"] = round(time.time() - t0, 1)
    rp = Path(report_path) if report_path else out / "report_cxp3.json"
    rp.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    print(f"ghi {rp}; tong {report['thoi_gian_giay']}s", flush=True)
    return report


def repro(cands: str, ref_cx130_path: str, fused_path: str, ref_path: str, report_path: str,
          delta: int = DELTA_REF, min_match: float = 0.99, wd: float = WD, b_cov: float = B_COV) -> dict:
    """Chạy công thức trên doc/ứng viên CŨ (doc nền = relevant_docs cx130cov, delta), so với
    relevant_chunks của submission_cx130cov. Trả report; exit 2 nếu tỷ lệ câu khớp < min_match."""
    t0 = time.time()
    ref130 = load_json(ref_cx130_path)
    doclist = {str(it["id"]): it["relevant_docs"] for it in ref130}
    want = {str(it["id"]): it["relevant_chunks"] for it in ref130}
    rr = {str(q): v for q, v in load_rerank(cands).items()}
    fu = {str(q): v for q, v in load_json(fused_path).items()}
    ref = load_json(ref_path)
    kq = {str(it["id"]): len(it["relevant_chunks"]) for it in ref}
    prep = prep_queries(rr, doclist, fu, {}, kq, wd, b_cov)
    del rr, fu
    sub, st = build_sub(sorted(doclist, key=lambda x: int(x) if str(x).isdigit() else x),
                        doclist, prep, {}, delta)
    n = n_ok = n_pos = n_pos_tot = 0
    vi_du = []
    for it in sub:
        q = str(it["id"])
        w, g = norm_pairs(want.get(q, [])), norm_pairs(it["relevant_chunks"])
        n += 1
        ok = g == w
        n_ok += ok
        for x, y in zip(g, w):
            n_pos_tot += 1
            n_pos += x == y
        n_pos_tot += abs(len(g) - len(w))
        if not ok and len(vi_du) < 5:
            i = next((j for j in range(max(len(g), len(w))) if g[j:j + 1] != w[j:j + 1]), None)
            vi_du.append({"id": q, "k_130": len(w), "got_n": len(g), "vi_tri_lech": i,
                          "got": g[i] if i is not None and i < len(g) else None,
                          "want": w[i] if i is not None and i < len(w) else None})
    rep = {"cau": n, "cau_khop": n_ok, "ty_le_cau_khop": n_ok / max(n, 1),
           "vi_tri_khop": n_pos, "ty_le_vi_tri_chunk_khop": n_pos / max(n_pos_tot, 1),
           "min_match": min_match, "ok": n_ok / max(n, 1) >= min_match,
           "delta": delta, "kprime": st["kprime"], "vi_du_lech": vi_du,
           "thoi_gian_giay": round(time.time() - t0, 1)}
    Path(report_path).write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    print(f"repro delta={delta}: {json.dumps(rep, ensure_ascii=False)}", flush=True)
    print(f"ghi {report_path}", flush=True)
    return rep


def check_zip(json_path: str, zip_path: str, report_path: str, name: str,
              max_zip: int = MAX_ZIP) -> int:
    """Đọc lại zip == JSON gốc + zip <= max_zip; cập nhật report. Exit 0/1/3/4."""
    p, zp = Path(json_path), Path(zip_path)
    st = dict(verify_zip_bytes(str(p), str(zp)))
    zb, jb = zp.stat().st_size, p.stat().st_size
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    duoi = zb <= max_zip
    rp = Path(report_path)
    report = json.loads(rp.read_text(encoding="utf-8")) if rp.exists() else {}
    d_used = (report.get("bai", {}).get(name, {}).get("delta", {}) or {}).get("dung")
    entry = {"name": name, "delta": d_used, "json_bytes": jb, "zip_bytes": zb,
             "duoi_gioi_han": duoi, "sha256": sha, "sha256_rut_gon": sha[:16],
             "bang_goc_bytes": st["bang_goc_bytes"], "bang_goc_json": st["bang_goc_json"],
             "entry": st["entry"], "ok": bool(st["ok"] and duoi)}
    report.setdefault("zip", {})[name] = entry
    rp.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    code = 0
    if not st["ok"]:
        code = 1
    elif not duoi:
        code = 4 if (d_used is None or d_used <= 0) else 3
        if code == 3:
            (p.parent / f"next_delta_{name}.txt").write_text(f"--delta {d_used - 1}", encoding="utf-8")
    print(json.dumps(entry, ensure_ascii=False), flush=True)
    return code


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    mq = sp.add_parser("mknq")
    mq.add_argument("--rank", required=True, help="cache_fhyde3.json (thứ hạng hợp nhất pool3)")
    mq.add_argument("--split", required=True, help="submission_split.json (n_q = số doc mỗi câu)")
    mq.add_argument("--new", default="", help="new_cands.json (đếm NEW lọt)")
    mq.add_argument("--out", required=True, help="doclist_p3_nq.json")
    mq.add_argument("--report", required=True)
    sd = sp.add_parser("swapdoc")
    sd.add_argument("--cx130", required=True, help="submission_cx130cov.json")
    sd.add_argument("--new", default="", help="new_cands.json (đếm TB NEW/câu)")
    sd.add_argument("--doclist", action="append", required=True, metavar="NAME=PATH")
    sd.add_argument("--outdir", required=True)
    rp = sp.add_parser("repro")
    rp.add_argument("--cands", required=True, help="ứng viên chunk cũ (chunk_dw_u900_ext.json.zst)")
    rp.add_argument("--ref-cx130cov", required=True)
    rp.add_argument("--fused", required=True, help="cache_rrf3d2.json")
    rp.add_argument("--ref-c120in", required=True, help="submission_c120in.json (k_q)")
    rp.add_argument("--report", required=True)
    rp.add_argument("--delta", type=int, default=DELTA_REF)
    rp.add_argument("--min-match", type=float, default=0.99)
    rp.add_argument("--wd", type=float, default=WD)
    rp.add_argument("--b-cov", type=float, default=B_COV)
    rn = sp.add_parser("run")
    rn.add_argument("--cands", required=True, help="ứng viên chunk mới (chunk_dw_u900_p3.json.zst)")
    rn.add_argument("--doclist", action="append", required=True, metavar="NAME=PATH")
    rn.add_argument("--fused", required=True, help="cache_rrf3d2.json (zd doc cũ)")
    rn.add_argument("--fused-new", default="", help="cache_rrf3d2_p3.json (zd doc NEW; rrf3d2 tai lap pool3)")
    rn.add_argument("--ref-c120in", required=True, help="submission_c120in.json (k_q)")
    rn.add_argument("--new-cands", default="", help="new_cands.json (đếm chunk từ doc NEW)")
    rn.add_argument("--outdir", required=True)
    rn.add_argument("--report", default="")
    rn.add_argument("--delta", type=int, default=None, help="ghi nhận delta (mặc định 57, giảm theo zip)")
    rn.add_argument("--delta-start", type=int, default=DELTA_REF)
    rn.add_argument("--max-zip", type=int, default=MAX_ZIP)
    rn.add_argument("--wd", type=float, default=WD)
    rn.add_argument("--b-cov", type=float, default=B_COV)
    cz = sp.add_parser("check-zip")
    cz.add_argument("--json", required=True)
    cz.add_argument("--zip", required=True)
    cz.add_argument("--report", required=True)
    cz.add_argument("--name", required=True)
    cz.add_argument("--max-zip", type=int, default=MAX_ZIP)
    a = ap.parse_args()
    if a.cmd == "mknq":
        mknq(a.rank, a.split, a.new, a.out, a.report)
        raise SystemExit(0)
    if a.cmd == "swapdoc":
        swapdoc(a.cx130, parse_doclists(a.doclist), a.outdir, a.new)
        raise SystemExit(0)
    if a.cmd == "repro":
        rep = repro(a.cands, a.ref_cx130cov, a.fused, a.ref_c120in, a.report,
                    a.delta, a.min_match, a.wd, a.b_cov)
        raise SystemExit(0 if rep["ok"] else 2)
    if a.cmd == "run":
        rep = run(a.cands, parse_doclists(a.doclist), a.fused, a.fused_new, a.ref_c120in,
                  a.outdir, a.new_cands, a.delta, a.delta_start, a.max_zip, a.wd, a.b_cov, a.report)
        over = [n for n, st in rep["bai"].items() if not st["uoc_zip"]["duoi_gioi_han"]]
        raise SystemExit(4 if over and all(rep["bai"][n]["delta"]["dung"] <= 0 for n in over) else
                         3 if over else 0)
    raise SystemExit(check_zip(a.json, a.zip, a.report, a.name, a.max_zip))


if __name__ == "__main__":
    main()
