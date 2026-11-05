"""vi26 = vi25_z99 chi doi do dai chunk (U=600 / U=450) theo luat vang moi 23/10.

Luat moi: chunk dung khi LCS/|g| >= 0,4 VA LCS/|c| >= 0,4 (|g| ~ 512 token, buoc 462). Chunk
U900 (trung vi 732 token) de bi phat boi ve LCS/|c|; chunk ngan hon thoa luat tot hon va chua
duoc nhieu chunk hon trong cung tran zip.

QUY UOC "CHI DOI MOT BIEN":
  - doclist + cong thuc chon chunk (diag14.order_q: s = z(logit chunk) + 4*z(rrf3d2), thuong
    b_cov = 1 cho chunk dau moi doc, k' = min(k_q + delta, |C_q|)) GIU NGUYEN - dung lai
    vi23.py chain voi pool/dense/hyde/q8b cua vi25.
  - tap ung vien chunk thay bang chunk U MOI dung lai TU CUNG nguon docwin (5 lop ghep giong
    vi23/vi24/vi25): top100 (docwin_top100 + chunkdocs100), ext (docwin_ext, loc D_q),
    v23 (docwin_p3 + docwin_vi23, loc D_q), vi24 (+p3reuse), vi25 (+p3reuse).
  - KHONG chay GPU moi; chi cat lai text + cham diem bang cua so docwin da co.
  - kep top-k cua lop top100 (goc --top-chunks 120 voi U=900) duoc noi theo ti le 900/U:
    U=600 -> 180, U=450 -> 240 (ngan sach byte zip la rang buoc chinh).

Stage (xem deploy/b660_vi26.bat):
  export-v23-docs  dung lai docs_vi23_dw_old/new da bi xoa staging (deterministic, cho lop v23)
  probe           in kich thuoc/loai du lieu nguon (nhanh, chay truc tiep SSH)
  cands           dung chunk_dw_u{U}_v25.json.zst trong diag26 (--units, mac dinh 900 600 450)
                  + so sanh U=900 voi chunk_dw_u900_v25.json.zst (kiem tra tai lap ung vien)
  sanity          chain U=900 delta=57 -> so submission_vi25_z99.json 1200/1200
  build           voi moi (ten, U): chain + zip LZMA voi delta LON NHAT thuc <= 99.000.000 B
                  + evaluate validate + stats
  report          bang bao cao cuoi (byte/sha256/delta/TB chunk/token/doc moi)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vi23 as core
from chunks_from_docwin import build_q_rows, load_json, select_q_rows, verify_verbatim
from dwx_merge import dq_docs_per_q, filter_new_rows, sort_rows, write_zst_stream
from rerank_run import load_docs
from vi24 import fingerprint, owned_lock, read, run, vi, write

ROOT = Path(os.environ.get("R2AI_ROOT", "D:/r2ai"))
W = ROOT / "data/sub/diag25"
B = ROOT / "data/sub/diag24"
V23 = ROOT / "data/sub/diag23"
V23O = V23 / "docwin_vi23"
P3 = ROOT / "data/sub/diag18/docwin_p3"
DW = ROOT / "data/sub/docwin"
DX = ROOT / "data/sub/docwin_ext"
C100 = ROOT / "data/sub/chunkdocs100"
OUT = ROOT / "data/sub/diag26"
POOL_RR = ROOT / "data/sub/cache_pool_rr.json"
Q8 = ROOT / "data/sub/qwen8b/cache_qwen8b_sorted.json"
REF_K = ROOT / "data/sub/diag14/submission_c120in.json"
REF_C = ROOT / "data/sub/diag11/chunks_rrf3d2_cf4.json"
REF_SUB = W / "submission_vi25_z99.json"
MAX_ZIP = 99_000_000
DELTA0 = 57
SANITY_DELTA = 57
UNITS = [900, 600, 450]
BASE_CAP = 120
ZIPS = [("vi26_u600", 600), ("vi26_u450", 450)]
ZIP_JSON_RATIO = 98_976_669 / 643_215_253


def cap_top(u: int) -> int:
    """Kep top-k cua lop top100 noi theo ti le 900/U (U=900 -> 120 nhu goc)."""
    return int(round(BASE_CAP * 900 / u))


def layer_specs(u: int) -> list:
    """5 lop ghep ung vien giong vi23/vi24/vi25, chi doi max_units = u."""
    p3_parts = [P3 / "docwin_p3.part000.jsonl.zst", P3 / "docwin_p3.part001.jsonl.zst"]
    return [
        {"name": "top100", "wins": [DW / "docwin_top100.json.zst"],
         "docs": [C100 / "chunkdocs100.part000.jsonl.zst", C100 / "chunkdocs100.part001.jsonl.zst"],
         "cands": C100 / "cands_chunkdocs100.json", "reuse": None, "iter": "wins",
         "top_docs": 100, "top_chunks": cap_top(u), "dq": False},
        {"name": "ext", "wins": [DX / "docwin_ext.json.zst"],
         "docs": [DX / "docwin_ext.part000.jsonl.zst", DX / "docwin_ext.part001.jsonl.zst"],
         "cands": DX / "cands_docwin_ext.json", "reuse": None, "iter": "wins",
         "top_docs": 500, "top_chunks": 100000, "dq": True},
        {"name": "v23", "wins": [P3 / "docwin_p3.json.zst", V23 / "docwin_vi23.json.zst"],
         "docs": [*p3_parts, *sorted(V23O.glob("docs_vi23_dw_old*.jsonl.zst")),
                  *sorted(V23O.glob("docs_vi23_dw_new*.jsonl.zst"))],
         "cands": V23O / "cands_docwin_v23all.json", "reuse": None, "iter": "wins",
         "top_docs": 500, "top_chunks": 100000, "dq": True},
        {"name": "v24", "wins": [B / "docwin_vi24.json.zst", P3 / "docwin_p3.json.zst"],
         "docs": [B / "docs_vi24_docwin.jsonl.zst", *sorted(B.glob("docs_vi24_old*.jsonl.zst")),
                  *p3_parts],
         "cands": B / "cands_docwin_vi24.json", "reuse": B / "cands_docwin_reuse.json",
         "iter": "cands",
         "top_docs": 500, "top_chunks": 100000, "dq": False},
        {"name": "v25", "wins": [W / "docwin_vi25.json.zst", P3 / "docwin_p3.json.zst"],
         "docs": [W / "docs_vi25_docwin.jsonl.zst", *sorted(W.glob("docs_vi25_old*.jsonl.zst")),
                  *p3_parts],
         "cands": W / "cands_docwin_vi25.json", "reuse": W / "cands_docwin_reuse.json",
         "iter": "cands",
         "top_docs": 500, "top_chunks": 100000, "dq": False},
    ]


def load_wins(spec: dict) -> dict:
    """{qid: {doc_id_str: wins}} - file sau ghi de file truoc (giong wins-merge / stage_chunk)."""
    wins = {}
    for wp in spec["wins"]:
        for q, m in load_json(str(wp)).items():
            wins.setdefault(str(q), {}).update({str(k): v for k, v in m.items()})
    return wins


def load_cands(spec: dict) -> dict:
    """{qid: [[doc_id, idx], ...]} - cap reuse duoc noi sau cap fresh (doc_rank = vi tri)."""
    cands = {str(q): [list(item) for item in v]
             for q, v in json.loads(Path(spec["cands"]).read_text(encoding="utf-8")).items()}
    if spec["reuse"]:
        for q, rows in json.loads(Path(spec["reuse"]).read_text(encoding="utf-8")).items():
            cands.setdefault(str(q), []).extend([list(item) for item in rows])
    return cands


def build_rows(spec: dict, u: int, pool: dict) -> tuple:
    """Chunk U cua mot lop: (rows {qid: [...]}, bao cao). Giong chunks_from_docwin --units u."""
    cands = load_cands(spec)
    wins = load_wins(spec)
    wanted = {item[0] for rows in cands.values() for item in rows}
    docs = load_docs(list(map(str, spec["docs"])), wanted=wanted)
    missing_text = len(wanted - set(docs))
    cache = {}
    out, n_raw = {}, 0
    absent = 0
    if spec["iter"] == "wins":
        todo = [(q, cands[q], wins_q) for q, wins_q in wins.items() if cands.get(q)]
    else:
        todo = [(q, rows, wins.get(q, {})) for q, rows in cands.items()]
    for q, rows, wins_q in todo:
        absent += sum(1 for item in rows
                      if str(item[0]) not in wins_q and docs.get(item[0], "").strip())
        rr = build_q_rows(rows, docs, wins_q, u, spec["top_docs"], cache)
        n_raw += len(rr)
        out[q] = select_q_rows(rr, spec["top_chunks"])
    kept = dropped = 0
    if spec["dq"]:
        out, kept, dropped = filter_new_rows(out, dq_docs_per_q(cands, pool, 100))
    else:
        kept = sum(len(r) for r in out.values())
    bad = verify_verbatim(out, docs)
    del docs
    return out, {"lop": spec["name"], "cap": len(wanted), "doc": len(cache), "dong_tho": n_raw,
                 "dong_giu": kept, "dong_bo_Dq": dropped, "thieu_text": missing_text,
                 "thieu_wins": absent, "khong_nguyen_van": bad}


def merge_into(merged: dict, rows: dict) -> int:
    """Gop rows vao merged (sort_rows giong merge_q); tra so dong trung (doc_id, chunk_idx)."""
    dup = 0
    for q, rr in rows.items():
        old = merged.get(q, [])
        if old:
            keys = {(r[0], r[2]) for r in old}
            dup += sum(1 for r in rr if (r[0], r[2]) in keys)
        merged[q] = sort_rows(old + rr)
    return dup


def stage_probe() -> None:
    for spec in layer_specs(900):
        cands = load_cands(spec)
        wins = load_wins(spec)
        n = sum(len(v) for v in cands.values())
        item = next((v[0] for v in cands.values() if v), None)
        print(json.dumps({"lop": spec["name"], "cau": len(cands), "cap": n,
                          "docs_file": [p.name for p in spec["docs"]],
                          "wins_cau": len(wins), "mau": item,
                          "loai_doc_id": type(item[0]).__name__ if item else None,
                          "top_chunks": spec["top_chunks"]}, ensure_ascii=False), flush=True)
    for p in [W / "chunk_dw_u900_v25.json.zst", OUT, V23O]:
        print(p, p.exists(), p.stat().st_size if p.exists() and p.is_file() else "", flush=True)


def stage_cands(units: list) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pool = json.loads(Path(POOL_RR).read_text(encoding="utf-8"))
    report = {}
    for u in units:
        t0 = time.time()
        merged = {}
        for spec in layer_specs(u):
            rows, rep = build_rows(spec, u, pool)
            rep["dong_trung_khoa"] = merge_into(merged, rows)
            print(json.dumps({"U": u, **rep}, ensure_ascii=False), flush=True)
        out = write_zst_stream(OUT / f"chunk_dw_u{u}_v25.json.zst", merged)
        n_row = sum(len(r) for r in merged.values())
        entry = {"U": u, "cau": len(merged), "dong": n_row, "TB_dong_cau": n_row / max(len(merged), 1),
                 "ghi": str(out), "bytes": out.stat().st_size, "giay": round(time.time() - t0, 1)}
        if u == 900:
            entry["so_voi_goc"] = cmp_cands(merged, read(W / "chunk_dw_u900_v25.json.zst"))
        report[str(u)] = entry
        print(json.dumps(entry, ensure_ascii=False), flush=True)
        del merged
    write(OUT / "cands_report.json", report)


def cmp_cands(got: dict, ref: dict) -> dict:
    """So ung vien U=900 dung lai voi file goc (phai trung tuyet doi de sanity co y nghia)."""
    n = n_eq = n_row = n_row_eq = 0
    for q in sorted(set(got) | set(ref), key=lambda x: int(x)):
        n += 1
        g, r = got.get(q, []), ref.get(q, [])
        n_eq += g == r
        n_row += max(len(g), len(r))
        n_row_eq += sum(1 for x, y in zip(g, r) if x == y)
    return {"cau": n, "cau_khop_y_nguyen": n_eq, "dong": n_row, "dong_khop": n_row_eq,
            "ok": n_eq == n}


def chain_args(cands, name: str, delta: int) -> list:
    return ["chain", "--mode", "that", "--name", name, "--pool", W / "cache_pool_v25.json",
            "--q8b", Q8, "--dense", W / "dense_v25_top200.json", "--hyde", W / "hyde_v25_top200.json",
            "--cands", cands, "--ref-kq", REF_K, "--ref-chunks", REF_C, "--delta", delta,
            "--out-dir", OUT]


def stage_sanity() -> None:
    name = "sanity900"
    vi(*chain_args(OUT / "chunk_dw_u900_v25.json.zst", name, SANITY_DELTA))
    cmp_rep = core.compare_subs(read(OUT / f"submission_{name}.json"), str(REF_SUB))
    write(OUT / "sanity_report.json", cmp_rep)
    print(json.dumps(cmp_rep, ensure_ascii=False), flush=True)


def chain_only(name: str, u: int, delta: int) -> dict:
    """chain (khong zip) de lay hist_kcap / json_bytes / tong_chunk cho uoc tinh delta."""
    vi(*chain_args(OUT / f"chunk_dw_u{u}_v25.json.zst", name, delta))
    rep = read(OUT / f"report_chain_{name}.json")
    return {"delta": delta, "json_bytes": rep.get("json_bytes"), "tong_chunk": rep.get("tong_chunk"),
            "hist_kcap": rep.get("hist_kcap"), "TB_chunk_cau": rep.get("TB_chunk_cau")}


def chain_zip_check(name: str, u: int, delta: int, max_zip: int) -> dict:
    """chain voi --delta, validate, zip LZMA, check_zip. Tra ket qua mot lan thu."""
    vi(*chain_args(OUT / f"chunk_dw_u{u}_v25.json.zst", name, delta))
    run("evaluate.py", "validate", OUT / f"submission_{name}.json")
    run("zip_submission.py", "--json", OUT / f"submission_{name}.json",
        "--zip", OUT / f"submission_{name}.zip", "--method", "lzma")
    code = core.check_zip(str(OUT / f"submission_{name}.json"), str(OUT / f"submission_{name}.zip"),
                          str(OUT / f"report_chain_{name}.json"), name, max_zip=max_zip)
    rep = read(OUT / f"report_chain_{name}.json")
    entry = rep.get("zip", {}).get(name, {})
    return {"code": code, "next_delta": entry.get("next_delta"), "zip_bytes": entry.get("zip_bytes"),
            "json_bytes": rep.get("json_bytes"), "tong_chunk": rep.get("tong_chunk"),
            "hist_kcap": rep.get("hist_kcap"), "delta": delta, "name": name}


def largest_delta(hist: list, json_now: float, n_now: int, max_zip: int, ratio: float) -> int:
    """delta lon nhat voi uoc zip = json * N(delta)/N_now * ratio <= max_zip (N = tong chunk)."""
    from cx130 import largest_delta_within
    return int(largest_delta_within(hist, json_now, n_now, 4000, max_zip, 1.0, ratio))


def find_delta(name: str, u: int, max_zip: int, delta_start: int, plus1: bool = True) -> tuple:
    """Delta LON NHAT voi zip LZMA THAT <= max_zip. Tra (delta, cac lan thu).

    Chain (khong zip) lay hist -> uoc delta bang ti le zip/json cua vi25_z99 -> zip that o delta do
    (ten that = bai nop); buoc +1 / next_delta xac nhan bang zip that (ten phu, khong de len bai).
    Mot delta da thu khong chay lai (tru khi can ghi bai o delta do).
    """
    probe = f"{name}p"
    base = chain_only(probe, u, delta_start)
    d0 = largest_delta(base["hist_kcap"], base["json_bytes"], base["tong_chunk"], max_zip, ZIP_JSON_RATIO)
    tried, finals = {}, set()

    def attempt(delta, final):
        if delta in tried and (not final or delta in finals):
            return tried[delta]
        tried[delta] = chain_zip_check(name if final else probe, u, delta, max_zip)
        if final:
            finals.add(delta)
        return tried[delta]

    cur = attempt(d0, True)
    if cur["code"] != 0:
        nxt = cur.get("next_delta")
        assert nxt is not None and 0 <= nxt < d0, cur
        cur = attempt(int(nxt), True)
    assert cur["code"] == 0, cur
    if plus1:
        for _ in range(2):
            ratio = (cur["zip_bytes"] / cur["json_bytes"]) if cur.get("json_bytes") else ZIP_JSON_RATIO
            up = largest_delta(cur["hist_kcap"], cur["json_bytes"], cur["tong_chunk"], max_zip, ratio)
            up = max(up, cur["delta"] + 1)
            if attempt(up, False)["code"] != 0:
                break
            r = attempt(up, True)
            if r["code"] != 0:
                break
            cur = r
    best = max((d for d in finals if tried[d]["code"] == 0), default=min(tried))
    return best, [{"delta": k, "code": v["code"], "zip_bytes": v["zip_bytes"],
                   "final": k in finals} for k, v in sorted(tried.items())]


def stage_build(zip_specs: list, delta_start: dict, plus1: bool = True) -> None:
    for name, u in zip_specs:
        delta, attempts = find_delta(name, u, MAX_ZIP, delta_start[name], plus1)
        same = read(OUT / f"doclist_{name}.json") == read(W / "doclist_vi25_z99.json")
        assert same, "doclist phai trung submission_vi25_z99.json"
        write(OUT / f"build_report_{name}.json",
              {"name": name, "U": u, "delta": delta, "attempts": attempts,
               "same_doclist": same, "plus1": plus1, "max_zip": MAX_ZIP,
               "zip": fingerprint(OUT / f"submission_{name}.zip")})
        print(json.dumps({"build": name, "delta": delta, "attempts": attempts,
                          "same_doclist": same}, ensure_ascii=False), flush=True)
    stage_report(zip_specs, {name: read(OUT / f"build_report_{name}.json")["delta"]
                             for name, _u in zip_specs})


def doc_token_lens(sub: list) -> list:
    from evaluate import tokenize
    return [len(tokenize(c["chunk_text"])) for it in sub for c in it["relevant_chunks"]]


def pct(vals: list, p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]


def stage_report(zip_specs: list, deltas: dict) -> None:
    table = []
    new_ids = set(map(str, read(W / "new_ids.json")))
    for name, u in zip_specs:
        sub = read(OUT / f"submission_{name}.json")
        fp = fingerprint(OUT / f"submission_{name}.zip")
        n = max(len(sub), 1)
        n_chunk = sum(len(it["relevant_chunks"]) for it in sub)
        lens = doc_token_lens(sub)
        table.append({
            "zip": f"submission_{name}.zip", "U": u, "delta": deltas.get(name),
            "bytes": fp["bytes"], "sha256": fp["sha256"],
            "TB_chunk_cau": round(n_chunk / n, 2), "tong_chunk": n_chunk,
            "TB_token_chunk": round(sum(lens) / max(len(lens), 1), 1),
            "p50_token_chunk": pct(lens, 0.5), "p95_token_chunk": pct(lens, 0.95),
            "TB_doc_khac_nhau_co_chunk_cau": round(
                sum(len({str(c["doc_id"]) for c in it["relevant_chunks"]}) for it in sub) / n, 2),
            "TB_chunk_tu_doc_moi_cau": round(
                sum(sum(1 for c in it["relevant_chunks"] if str(c["doc_id"]) in new_ids)
                    for it in sub) / n, 2)})
        vi("stats", "--submission", OUT / f"submission_{name}.json", "--fhyde", W / "cache_fhyde_v25.json",
           "--fhyde-old", B / "cache_fhyde_v24.json", "--pool", W / "cache_pool_v25.json",
           "--new-ids", W / "new_ids.json", "--links", W / "new_host.json",
           "--report", OUT / f"stats_report_{name}.json")
    write(OUT / "zip_table.json", table)
    for row in table:
        print(json.dumps(row, ensure_ascii=False), flush=True)


def stage_report_only(zip_specs: list) -> None:
    deltas = {}
    for name, _u in zip_specs:
        rep = read(OUT / f"build_report_{name}.json") or {}
        deltas[name] = rep.get("delta")
    stage_report(zip_specs, deltas)


def stage_export_v23_docs() -> None:
    """Tao lai docs_vi23_dw_old/new (bi xoa staging) bang lenh goc - deterministic."""
    run("export_docs.py", "--index", ROOT / "data/index/vi", "--cache", V23O / "cache_docwin_old.json",
        "--n", "1000", "--chars", "200000", "--no-title", "--part-mb", "80",
        "--out-docs", V23O / "docs_vi23_dw_old.jsonl.zst", "--out-cands", V23O / "cands_from_export_old.json")
    run("vi23.py", "export-new", "--shards", V23 / "shards", "--bp", ROOT / "data/index/vi23/boilerplate.json",
        "--idx", V23 / "new_idx.json", "--mode", "docwin", "--out", V23O / "docs_vi23_dw_new.jsonl.zst",
        "--report", V23O / "export_dw_new_report.json")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["probe", "cands", "sanity", "build", "report",
                                          "export-v23-docs"])
    parser.add_argument("--units", type=int, nargs="*", default=UNITS)
    parser.add_argument("--delta-start", type=int, nargs="*", default=None,
                        help="delta khoi tinh cho tung zip (thu tu nhu ZIPS)")
    parser.add_argument("--zips", type=int, nargs="*", default=None,
                        help="chi build zip theo chi so trong ZIPS (0 = u600, 1 = u450)")
    parser.add_argument("--no-plus1", action="store_true", help="bo buoc +1 xac nhan delta lon nhat")
    args = parser.parse_args()
    specs = [ZIPS[i] for i in (args.zips if args.zips is not None else range(len(ZIPS)))]
    starts = {}
    default_start = {"vi26_u600": 120, "vi26_u450": 187}
    for i, (name, _u) in enumerate(specs):
        starts[name] = (args.delta_start[i] if args.delta_start and i < len(args.delta_start)
                        else default_start[name])
    OUT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "locks" / "heavy_job.lock"
    with owned_lock(lock, f"vi26 {args.stage} pid={os.getpid()}"):
        if args.stage == "probe":
            stage_probe()
        elif args.stage == "cands":
            stage_cands(args.units)
        elif args.stage == "sanity":
            stage_sanity()
        elif args.stage == "build":
            stage_build(specs, starts, not args.no_plus1)
        elif args.stage == "report":
            stage_report_only(specs)
        else:
            stage_export_v23_docs()
    print("DONE", args.stage, flush=True)


if __name__ == "__main__":
    main()
