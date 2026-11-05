"""vi27 = vi25_z99 chi doi b_cov (2.0 / 4.0): thuong phu doc de tang Chunks R.

Giay thuyet: recall chunk bi chan boi SO DOC co chunk (moi doc vang ~1 chunk vang); tang
thuong b cho chunk dau moi doc (apply_bonus cx93) se phu them doc truoc khi doc khac nhan
chunk thu 2. b = inf da lo (08/10) -> thu muc do vua phai 2.0 / 4.0.

QUY UOC "CHI DOI MOT BIEN":
  - CUNG doclist (diag25/doclist_vi25_z99.json), CUNG ung vien chunk
    (diag26/chunk_dw_u900_v25.json.zst, da kiem trung tuyet doi voi ban goc), CUNG delta 57.
  - k' = min(k_q + delta, |C_q|) khong phu thuoc b_cov -> tong chunk giu nguyen.
  - chi doi --b-cov: vi27_b2 = 2.0, vi27_b4 = 4.0.
  - zip LZMA that > 99.000.000 B thi ha delta tung buoc nhat (1) cho vua, ghi lai attempts.

Stage (xem deploy/b660_vi27.bat):
  sanity  chain b_cov=1.0 delta=57 -> phai trung submission_vi25_z99.json 1200/1200 (doc+chunk)
  build   voi moi (ten, b): chain --b-cov b --delta 57 -> evaluate validate -> zip LZMA
          -> check_zip <= 99.000.000 + assert doclist trung diag25/doclist_vi25_z99.json
  report  bang bao cao (kem cot z99 tham chieu): bytes, sha256, delta, b_cov, TB chunk/cau,
          TB doc khac nhau co chunk/cau, TB chunk/doc co chunk, so chunk khac z99 (TB/cau),
          % chunk trung z99, phan bo hang fhyde cua doc moi duoc phu them
"""
import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vi23 as core
from vi24 import fingerprint, owned_lock, read, run, vi, write

ROOT = Path(os.environ.get("R2AI_ROOT", "D:/r2ai"))
W = ROOT / "data/sub/diag25"
C26 = ROOT / "data/sub/diag26"
OUT = ROOT / "data/sub/diag27"
Q8 = ROOT / "data/sub/qwen8b/cache_qwen8b_sorted.json"
REF_K = ROOT / "data/sub/diag14/submission_c120in.json"
REF_C = ROOT / "data/sub/diag11/chunks_rrf3d2_cf4.json"
REF_SUB = W / "submission_vi25_z99.json"
REF_DOC = W / "doclist_vi25_z99.json"
REF_ZIP = W / "submission_vi25_z99.zip"
CANDS = C26 / "chunk_dw_u900_v25.json.zst"
FHYDE = OUT / "cache_fhyde_that.json"
FHYDE_REF = W / "cache_fhyde_v25.json"
MAX_ZIP = 99_000_000
DELTA = 57
SANITY_DELTA = 57
SANITY_CAU = 1200
MAX_STEPDOWN = 10
ZIPS = [("vi27_b2", 2.0), ("vi27_b4", 4.0)]
RANK_BANDS = (("1-50", 1, 50), ("51-100", 51, 100), ("101-165", 101, 165), ("166+", 166, None))
RANK_NONE = "khong_co_hang"


def chain_args(name: str, delta: int, b_cov: float) -> list:
    """Lenh vi23 chain giong vi25_z99 (cung pool/dense/hyde/q8b/ref) + --b-cov, out-dir diag27."""
    return ["chain", "--mode", "that", "--name", name,
            "--pool", W / "cache_pool_v25.json", "--q8b", Q8,
            "--dense", W / "dense_v25_top200.json", "--hyde", W / "hyde_v25_top200.json",
            "--cands", CANDS, "--ref-kq", REF_K, "--ref-chunks", REF_C,
            "--delta", delta, "--b-cov", b_cov, "--out-dir", OUT]


def require_sanity(cmp_rep: dict) -> None:
    """Sanity phai khop 1200/1200 ca doc lan chunk, khong thi dung (exit 2)."""
    ok = (cmp_rep.get("ok") and cmp_rep.get("so_cau") == SANITY_CAU
          and cmp_rep.get("cau_khop_doc") == SANITY_CAU
          and cmp_rep.get("cau_khop_chunk") == SANITY_CAU)
    if not ok:
        print("SANITY FAIL", json.dumps(cmp_rep, ensure_ascii=False), flush=True)
        raise SystemExit(2)


def stage_sanity() -> None:
    name = "sanity"
    vi(*chain_args(name, SANITY_DELTA, 1.0))
    cmp_rep = core.compare_subs(read(OUT / f"submission_{name}.json"), str(REF_SUB))
    write(OUT / "sanity_report.json", cmp_rep)
    print(json.dumps(cmp_rep, ensure_ascii=False), flush=True)
    require_sanity(cmp_rep)


def chain_zip_check(name: str, b_cov: float, delta: int, max_zip: int) -> dict:
    """chain + evaluate validate + zip LZMA + check_zip. Tra ket qua mot lan thu."""
    vi(*chain_args(name, delta, b_cov))
    run("evaluate.py", "validate", OUT / f"submission_{name}.json")
    run("zip_submission.py", "--json", OUT / f"submission_{name}.json",
        "--zip", OUT / f"submission_{name}.zip", "--method", "lzma")
    code = core.check_zip(str(OUT / f"submission_{name}.json"), str(OUT / f"submission_{name}.zip"),
                          str(OUT / f"report_chain_{name}.json"), name, max_zip=max_zip)
    rep = read(OUT / f"report_chain_{name}.json")
    entry = rep.get("zip", {}).get(name, {})
    return {"code": code, "zip_bytes": entry.get("zip_bytes"), "json_bytes": rep.get("json_bytes"),
            "tong_chunk": rep.get("tong_chunk"), "delta": delta, "name": name}


def find_delta(name: str, b_cov: float, max_zip: int, start: int) -> tuple:
    """Delta voi zip LZMA that <= max_zip. Neu vuot thi ha delta tung buoc nhat (1 buoc/lan).

    Tra (delta, attempts). code 1 (noi dung zip khac JSON) la loi nghiem trong -> dung.
    """
    attempts, delta = [], start
    for _ in range(MAX_STEPDOWN + 1):
        r = chain_zip_check(name, b_cov, delta, max_zip)
        attempts.append({"delta": delta, "code": r["code"], "zip_bytes": r["zip_bytes"]})
        if r["code"] == 0:
            return delta, attempts
        if r["code"] == 1:
            raise RuntimeError(f"zip khac JSON goc: {name} delta={delta}")
        delta -= 1
        if delta < 0:
            break
    raise RuntimeError(f"khong tim duoc delta voi zip <= {max_zip}: {name} attempts={attempts}")


def stage_build(zip_specs: list, starts: dict) -> None:
    for name, b in zip_specs:
        t0 = time.time()
        delta, attempts = find_delta(name, b, MAX_ZIP, starts[name])
        same = read(OUT / f"doclist_{name}.json") == read(REF_DOC)
        assert same, f"doclist_{name} phai trung doclist_vi25_z99.json"
        write(OUT / f"build_report_{name}.json",
              {"name": name, "b_cov": b, "delta": delta, "attempts": attempts,
               "same_doclist": same, "max_zip": MAX_ZIP, "giay": round(time.time() - t0, 1),
               "zip": fingerprint(OUT / f"submission_{name}.zip")})
        print(json.dumps({"build": name, "b_cov": b, "delta": delta, "attempts": attempts,
                          "same_doclist": same}, ensure_ascii=False), flush=True)
    stage_report_only(zip_specs)


def chunk_pairs(item: dict) -> set:
    return {(str(c["doc_id"]), c["chunk_text"]) for c in item["relevant_chunks"]}


def covered_docs(item: dict) -> set:
    return {str(c["doc_id"]) for c in item["relevant_chunks"]}


def rank_band(rank) -> str:
    if rank is None:
        return RANK_NONE
    for name, lo, hi in RANK_BANDS:
        if lo <= rank and (hi is None or rank <= hi):
            return name
    return RANK_NONE


def fhyde_rank_maps(path) -> dict:
    """{qid: {doc_id_str: hang}} voi hang = vi tri 1-based trong hit list fhyde (dedup doc)."""
    out = {}
    for q, hits in read(path).items():
        out[str(q)] = {str(d): i for i, d in enumerate(core.hit_ids(hits, None), 1)}
    return out


def sub_metrics(sub: list, ref_pairs: dict, ref_docs: dict, rank_maps: dict) -> dict:
    """So moi cau voi z99: cap (doc_id, chunk_text) cung cau; phan bo hang fhyde doc moi."""
    n = max(len(sub), 1)
    n_chunk = cap_doc = match = khac = n_moi = 0
    docs_per_q, bands = [], Counter()
    for it in sub:
        q = str(it["id"])
        pairs = [(str(c["doc_id"]), c["chunk_text"]) for c in it["relevant_chunks"]]
        docs = {p[0] for p in pairs}
        rp, rd = ref_pairs.get(q, set()), ref_docs.get(q, set())
        n_chunk += len(pairs)
        cap_doc += len(docs)
        docs_per_q.append(len(docs))
        match += sum(1 for p in set(pairs) if p in rp)
        khac += sum(1 for p in set(pairs) if p not in rp)
        for d in docs - rd:
            n_moi += 1
            bands[rank_band(rank_maps.get(q, {}).get(d))] += 1
    hang = {name: bands.get(name, 0) for name, _lo, _hi in RANK_BANDS}
    hang[RANK_NONE] = bands.get(RANK_NONE, 0)
    return {"tong_chunk": n_chunk, "TB_chunk_cau": round(n_chunk / n, 2),
            "TB_doc_khac_nhau_co_chunk_cau": round(sum(docs_per_q) / n, 2),
            "cap_cau_doc_co_chunk": cap_doc,
            "TB_chunk_moi_doc_co_chunk": round(n_chunk / max(cap_doc, 1), 3),
            "TB_chunk_khac_z99_cau": round(khac / n, 2),
            "pct_chunk_trung_z99": round(100 * match / max(n_chunk, 1), 2),
            "so_cap_doc_moi_phu": n_moi, "hang_fhyde_doc_moi": hang}


def stage_report(zip_specs: list, deltas: dict) -> None:
    ref = read(REF_SUB)
    ref_pairs = {str(it["id"]): chunk_pairs(it) for it in ref}
    ref_docs = {str(it["id"]): covered_docs(it) for it in ref}
    rank_maps = fhyde_rank_maps(FHYDE if FHYDE.exists() else FHYDE_REF)
    rows = []
    fp = fingerprint(REF_ZIP)
    rows.append({"zip": REF_ZIP.name, "name": "vi25_z99", "b_cov": 1.0, "delta": DELTA,
                 "bytes": fp["bytes"], "sha256": fp["sha256"],
                 **sub_metrics(ref, ref_pairs, ref_docs, rank_maps)})
    for name, b in zip_specs:
        sub = read(OUT / f"submission_{name}.json")
        fp = fingerprint(OUT / f"submission_{name}.zip")
        rows.append({"zip": f"submission_{name}.zip", "name": name, "b_cov": b,
                     "delta": deltas.get(name), "bytes": fp["bytes"], "sha256": fp["sha256"],
                     **sub_metrics(sub, ref_pairs, ref_docs, rank_maps)})
        del sub
    write(OUT / "zip_table.json", rows)
    for row in rows:
        print(json.dumps(row, ensure_ascii=False), flush=True)


def stage_report_only(zip_specs: list) -> None:
    deltas = {}
    for name, _b in zip_specs:
        rep = read(OUT / f"build_report_{name}.json") if (OUT / f"build_report_{name}.json").exists() else {}
        deltas[name] = rep.get("delta") if rep and rep.get("delta") is not None \
            else (read(OUT / f"report_chain_{name}.json") or {}).get("delta")
    stage_report(zip_specs, deltas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["sanity", "build", "report"])
    parser.add_argument("--zips", type=int, nargs="*", default=None,
                        help="chi build/report zip theo chi so trong ZIPS (0 = b2, 1 = b4)")
    parser.add_argument("--delta-start", type=int, nargs="*", default=None,
                        help="delta khoi tinh cho tung zip (thu tu nhu ZIPS, mac dinh 57)")
    args = parser.parse_args()
    specs = [ZIPS[i] for i in (args.zips if args.zips is not None else range(len(ZIPS)))]
    starts = {}
    for i, (name, _b) in enumerate(specs):
        starts[name] = args.delta_start[i] if args.delta_start and i < len(args.delta_start) else DELTA
    OUT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / "locks" / "heavy_job.lock"
    with owned_lock(lock, f"vi27 {args.stage} pid={os.getpid()}"):
        if args.stage == "sanity":
            stage_sanity()
        elif args.stage == "build":
            stage_build(specs, starts)
        else:
            stage_report_only(specs)
    print("DONE", args.stage, flush=True)


if __name__ == "__main__":
    main()
