"""vi28 = vi25_z99 (#6757) - 16.079 ban ghi nhiem cheo stale-tab (24/10).

Crawler phat hien 16.079 ban ghi trong 3 goi da ban giao (handoff_vi_2026{1017,1021,1023})
mang text cua trang khac (loi stale-tab); goi thay the *_fix (96.908 rec) da duoc giao
(docs/crawl/HANDOFF.md muc DINH CHINH 24/10). vi28 la PHEP LOC tren chain vi25_z99: gom
16.079 id trong audit/contamination_bad_ids_20261024.tsv khoi MOI dau vao chain (pool, q8b,
dense, hyde, ung vien chunk, ref_chunks), giu nguyen thu tu + diem phan con lai (khong lap
cho trong). Khong gỡ 445 ban trung canonical, khong gỡ 13 id false positive. Khong dung z87,
khong sua artefact vi25/diag25 (chi doc), khong nop bai.

Kiem giay thuyet (stage verify): goi _fix = goi cu - 16.079 id nhiem - ~445 ban trung
canonical, text cac id con lai giong het. |fix - cu| > 0 hoac text hash khac -> DUNG
(can z87 nhung/rerank lai, ngoai pham vi dem nay).

Stage (xem deploy/b660_vi28.bat):
  verify  doc bad/fp ids + 6 goi handoff (id + sha256(text)) + ids.npy chi muc vi25
          -> verify_report.json + bad_ids.json; anh huong #6757 tren submission_vi25_z99
          (cap (q, doc) bad, chunk cua doc bad, bad id trong ref_chunks/c120in).
          Lech giay thuyet -> exit 2, khong build.
  sanity  chain voi tap loai RONG, delta 57 -> phai tai lap submission_vi25_z99.json
          1200/1200 (doc + chunk, dung thu tu).
  build   loc moi dau vao bang 16.079 bad id -> find_delta (zip LZMA THAT <= 99.000.000 B,
          khoi dau 57) -> chain vi28 + evaluate validate + zip + check_zip; assert 0 bad id
          trong relevant_docs/relevant_chunks, moi chunk nam trong doc list, 1.200 cau.
  report  bang so voi z99: TB doc/cau, n_q TB, TB chunk/cau, doc co chunk/cau, delta, byte
          zip, sha256, so cau doc thay doi, top-30 doc trung z99.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vi23 as core
from vi24 import fingerprint, owned_lock, read, run, vi, write

ROOT = Path(os.environ.get("R2AI_ROOT", "D:/r2ai"))
W = ROOT / "data/sub/diag25"                    # vi25 - CHI DOC
OUT = ROOT / "data/sub/diag28"
IN0 = OUT / "in0"                               # dau vao loc voi tap loai RONG (sanity)
IN1 = OUT / "in1"                               # dau vao loc 16.079 bad id (build)
AUDIT = ROOT / "data/crawl/async_recovery_20261009/audit"
CRAWL = ROOT / "data/crawl"
BAD_TSV = AUDIT / "contamination_bad_ids_20261024.tsv"
FP_TSV = AUDIT / "contamination_false_positive_20261024.tsv"
PKG_TAGS = ["20261017", "20261021", "20261023"]
IDX_IDS = ROOT / "data/index/vi25/ids.npy"
Q8 = ROOT / "data/sub/qwen8b/cache_qwen8b_sorted.json"
REF_K = ROOT / "data/sub/diag14/submission_c120in.json"
REF_C = ROOT / "data/sub/diag11/chunks_rrf3d2_cf4.json"
REF_SUB = W / "submission_vi25_z99.json"
REF_ZIP = W / "submission_vi25_z99.zip"
NAME = "vi28"
SANITY_NAME = "vi28_sanity"
PROBE = "vi28p"
SANITY_DELTA = 57
DELTA0 = 57
MAX_ZIP = 99_000_000
THR, LO, HI, WD, B_COV = -2.0, 40, 400, 4.0, 1.0
FILE_NAMES = ("pool_v28.json", "q8b_v28.json", "dense_v28.json", "hyde_v28.json",
              "cands_v28.json.zst", "ref_chunks_v28.json")


# ---------------------------------------------------------------- tien ich loc

def read_ids_tsv(path) -> list:
    """id (cot dau) cua file TSV; bo header (dong dau khong phai so) va dong rong."""
    out = []
    for i, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines()):
        line = line.strip()
        if not line:
            continue
        first = line.split("\t")[0].strip()
        if not first.lstrip("-").isdigit():
            if i == 0 or not out:
                continue
            raise ValueError(f"dong khong phai id: {path} {line[:40]}")
        out.append(int(first))
    return out


def scan_package(pkg_dir) -> dict:
    """{id: sha256(text)} cua mot goi handoff (*.jsonl.zst), kem so ban ghi/trung id."""
    ids, records, trung_khac = {}, 0, 0
    files = [p.name for p in sorted(Path(pkg_dir).glob("*.jsonl.zst"))]
    for name in files:
        for r in core.read_shard_records(Path(pkg_dir) / name):
            records += 1
            did = int(r["id"])
            h = hashlib.sha256((r.get("text") or "").encode("utf-8")).hexdigest()
            if did in ids and ids[did] != h:
                trung_khac += 1
            ids[did] = h
    return {"ids": ids, "records": records, "files": files, "trung_id_khac_text": trung_khac}


def hypothesis_report(bad: set, fp: set, old: dict, fix: dict, idx_ids: set) -> dict:
    """So lieu kiem giay thuyet goi _fix = goi cu - bad - trung canonical, text giong het.

    old/fix = {tag: {id: sha256(text)}}. ok_thuyet = fix khong co id ngoai goi cu VA text
    khong id nao khac -> tiep tuc build duoc bang LOC tren b660 (khong can z87).
    """
    per, tong = {}, {}
    for tag in sorted(old):
        o, f = old[tag], fix[tag]
        common = set(f) & set(o)
        row = {"so_id_cu": len(o), "so_id_fix": len(f),
               "bad_trong_cu": len(bad & set(o)), "bad_trong_fix": len(bad & set(f)),
               "fix_ngoai_cu": len(set(f) - set(o)),
               "cu_thua_sau_fix": len(set(o) - set(f) - bad),
               "text_hash_khac": sum(1 for i in common if f[i] != o[i])}
        per[tag] = row
        for k in row:
            tong[k] = tong.get(k, 0) + row[k]
    old_u = set().union(*(set(v) for v in old.values()))
    fix_u = set().union(*(set(v) for v in fix.values()))
    text_khac = tong.get("text_hash_khac", 0)
    fix_ngoai = tong.get("fix_ngoai_cu", 0)
    ok = fix_ngoai == 0 and text_khac == 0
    return {"so_dong_bad": len(bad), "so_dong_fp": len(fp), "fp_trong_bad": len(fp & bad),
            "theo_goi": per, "tong": dict(tong),
            "bad_trong_cu_gop": len(bad & old_u), "bad_trong_fix_gop": len(bad & fix_u),
            "fix_ngoai_cu_gop": len(fix_u - old_u),
            "cu_thua_sau_fix_gop": len(old_u - fix_u - bad),
            "text_hash_khac_gop": text_khac,
            "bad_trong_idx_vi25": len(bad & set(idx_ids)),
            "ok_thuyet": ok,
            "ket_luan": "fix = cu - bad - trung canonical, text giong het; loc tren CPU du"
                        if ok else "LECH giay thuyet: can z87 nhung/rerank lai, DUNG khong build"}


def impact_report(sub: list, bad: set, ref_chunks: list, c120in: list) -> dict:
    """Anh huong cua 16.079 bad id den #6757 (vi25_z99) + ref_chunks + c120in."""
    bs = {str(i) for i in bad}
    n = max(len(sub), 1)
    cap = chunk_bad = cau_anh = 0
    for it in sub:
        hit = sum(1 for d in {str(x) for x in it["relevant_docs"]} if d in bs)
        cap += hit
        cau_anh += bool(hit)
        chunk_bad += sum(1 for c in it["relevant_chunks"] if str(c["doc_id"]) in bs)

    def pair_count(items):
        n_pair, ids = 0, set()
        for it in items or []:
            seen = {str(c["doc_id"]) for c in it.get("relevant_chunks", [])} & bs
            n_pair += len(seen)
            ids |= seen
        return n_pair, ids

    ref_pair, ref_ids = pair_count(ref_chunks)
    c120_pair, c120_ids = pair_count(c120in)
    for it in c120in or []:
        c120_ids |= {str(d) for d in it.get("relevant_docs", [])} & bs
    return {"so_cap_qdoc_bad_trong_bai": cap, "TB_qdoc_bad_cau": round(cap / n, 2),
            "so_cau_anh_huong": cau_anh, "so_chunk_cua_doc_bad": chunk_bad,
            "ref_chunks": {"so_cap_qdoc_bad": ref_pair, "so_id_bad": len(ref_ids)},
            "c120in": {"so_cap_qdoc_bad": c120_pair, "so_id_bad": len(c120_ids)}}


# ---------------------------------------------------------------- loc dau vao chain

def filter_hits(hits: list, bad: set) -> list:
    """Bo hit co doc_id trong bad; giu nguyen thu tu + diem phan con lai (khong lap cho)."""
    bs = {str(i) for i in bad}
    return [h for h in hits or [] if str(h[0]) not in bs]


def filter_cache(cache: dict, bad: set) -> dict:
    """{qid: hits} sau khi bo hit bad (giu nguyen key + thu tu)."""
    bs = {str(i) for i in bad}
    return {str(q): [h for h in hits or [] if str(h[0]) not in bs] for q, hits in cache.items()}


def filter_cands(cands: dict, bad: set) -> dict:
    """{qid: [doc_id, doc_rank, chunk_idx, logit, chunk_text]} bo dong chunk cua doc bad."""
    bs = {str(i) for i in bad}
    return {str(q): [r for r in rows if str(r[0]) not in bs] for q, rows in cands.items()}


def filter_ref_chunks(ref: list, bad: set) -> list:
    """ref_chunks (list {id, relevant_chunks}) bo chunk cua doc bad; giu nguyen cau + thu tu."""
    bs = {str(i) for i in bad}
    out = []
    for it in ref or []:
        kept = [c for c in it.get("relevant_chunks", []) if str(c["doc_id"]) not in bs]
        out.append({**it, "relevant_chunks": kept})
    return out


def write_zst(path, obj) -> int:
    import zstandard
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    with Path(path).open("wb") as f:
        with zstandard.ZstdCompressor(level=6).stream_writer(f) as z:
            z.write(raw)
    return len(raw)


def write_inputs(bad: set, indir: Path) -> dict:
    """Ghi 6 dau vao chain da loc theo bad vao indir; tra so lieu da gỡ (de bao cao)."""
    indir.mkdir(parents=True, exist_ok=True)
    spec = [("pool_v28.json", W / "cache_pool_v25.json", "cache"),
            ("q8b_v28.json", Q8, "cache"),
            ("dense_v28.json", W / "dense_v25_top200.json", "cache"),
            ("hyde_v28.json", W / "hyde_v25_top200.json", "cache"),
            ("cands_v28.json.zst", W / "chunk_dw_u900_v25.json.zst", "cands"),
            ("ref_chunks_v28.json", REF_C, "ref")]
    rep = {"bad_id": len(bad), "out_dir": str(indir), "files": {}}
    for name, src, kind in spec:
        obj = read(src)
        if kind == "cache":
            kept = filter_cache(obj, bad)
            n_before = sum(len(v) for v in obj.values())
            n_after = sum(len(v) for v in kept.values())
            out_obj = kept
        elif kind == "cands":
            out_obj = filter_cands(obj, bad)
            n_before = sum(len(v) for v in obj.values())
            n_after = sum(len(v) for v in out_obj.values())
        else:
            out_obj = filter_ref_chunks(obj, bad)
            n_before = sum(len(it.get("relevant_chunks", [])) for it in obj)
            n_after = sum(len(it["relevant_chunks"]) for it in out_obj)
        dest = indir / name
        if name.endswith(".zst"):
            nbytes = write_zst(dest, out_obj)
        else:
            nbytes = write(dest, out_obj)
        rep["files"][name] = {"nguon": str(src), "truoc": n_before, "sau": n_after,
                              "bo": n_before - n_after, "json_bytes": nbytes}
        del obj, out_obj
    return rep


def load_bad_ids() -> set:
    """Tap 16.079 id se gỡ: bad_ids.json (verify da ghi) hoac doc truc tiep TSV."""
    p = OUT / "bad_ids.json"
    return set(map(int, read(p))) if p.exists() else set(read_ids_tsv(BAD_TSV))


# ---------------------------------------------------------------- chain + build

def chain_args(name: str, delta: int, indir: Path) -> list:
    """Lenh vi23 chain giong vi25_z99 (cung tham so) nhung dau vao da loc trong indir."""
    return ["chain", "--mode", "that", "--name", name,
            "--pool", indir / "pool_v28.json", "--q8b", indir / "q8b_v28.json",
            "--dense", indir / "dense_v28.json", "--hyde", indir / "hyde_v28.json",
            "--cands", indir / "cands_v28.json.zst", "--ref-kq", REF_K,
            "--ref-chunks", indir / "ref_chunks_v28.json",
            "--thr", THR, "--lo", LO, "--hi", HI, "--wd", WD, "--b-cov", B_COV,
            "--delta", delta, "--out-dir", OUT]


def require_sanity(cmp_rep: dict) -> None:
    """Sanity phai khop 1200/1200 ca doc lan chunk, khong thi dung (exit 2)."""
    ok = (cmp_rep.get("ok") and cmp_rep.get("so_cau") == 1200
          and cmp_rep.get("cau_khop_doc") == 1200
          and cmp_rep.get("cau_khop_chunk") == 1200)
    if not ok:
        print("SANITY FAIL", json.dumps(cmp_rep, ensure_ascii=False), flush=True)
        raise SystemExit(2)


def chain_zip_check(name: str, delta: int, max_zip: int, indir: Path, validate: bool = False) -> dict:
    """Chain voi --delta, (tuy chon) evaluate validate, zip LZMA that, check_zip. Mot lan thu."""
    vi(*chain_args(name, delta, indir))
    if validate:
        run("evaluate.py", "validate", OUT / f"submission_{name}.json")
    run("zip_submission.py", "--json", OUT / f"submission_{name}.json",
        "--zip", OUT / f"submission_{name}.zip", "--method", "lzma")
    code = core.check_zip(str(OUT / f"submission_{name}.json"), str(OUT / f"submission_{name}.zip"),
                          str(OUT / f"report_chain_{name}.json"), name, max_zip=max_zip)
    rep = read(OUT / f"report_chain_{name}.json")
    entry = rep.get("zip", {}).get(name, {})
    return {"code": code, "next_delta": entry.get("next_delta"), "zip_bytes": entry.get("zip_bytes"),
            "json_bytes": rep.get("json_bytes"), "tong_chunk": rep.get("tong_chunk"),
            "hist_kcap": rep.get("hist_kcap"), "n_q_TB": rep.get("n_q_TB")}


def find_delta(indir: Path, max_zip: int, delta0: int, probe: str) -> tuple:
    """Delta LON NHAT voi zip LZMA THAT <= max_zip (giong vi25.find_delta; cache theo lan thu).

    Giam bang vong check-zip/next_delta den khi duoi tran, roi uoc luong cx130.largest_delta_within
    + nhi phan/+1 xac nhan bang zip that (vi28 co the tang delta vi bo chunk). Tong chunk khong
    tang nua = bao hoa |C_q| -> dung o delta do. Ket qua moi delta duoc cache de resume.
    """
    from cx130 import largest_delta_within
    cache_file = OUT / f"attempts_{probe}.json"
    tried = {int(k): v for k, v in (read(cache_file) if cache_file.exists() else {}).items()}

    def attempt(delta):
        if delta not in tried:
            tried[delta] = chain_zip_check(probe, delta, max_zip, indir)
            write(cache_file, {str(k): v for k, v in sorted(tried.items())})
            print("ATTEMPT", json.dumps({k: v for k, v in tried[delta].items()
                                         if k != "hist_kcap"}, ensure_ascii=False), flush=True)
        return tried[delta]

    def fits(delta):
        return attempt(delta)["code"] == 0

    def grew(delta, base):
        return tried[delta].get("tong_chunk") != tried[base].get("tong_chunk")

    lo = delta0
    while not fits(lo):
        r = tried[lo]
        assert r["code"] == 3 and r["next_delta"] is not None and 0 <= r["next_delta"] < lo, r
        lo = r["next_delta"]
    hist = next(iter(tried.values()))["hist_kcap"]
    for _ in range(3):
        r = tried[lo]
        up = int(largest_delta_within(hist, r["json_bytes"], r["tong_chunk"], lo + 500, max_zip)) \
            if hist and r.get("tong_chunk") and r.get("json_bytes") else lo + 1
        up = max(up, lo + 1)
        if fits(up) and grew(up, lo):
            lo = up
            continue
        if fits(up):
            break  # bao hoa |C_q|: delta lon hon van cho cung noi dung, dung o lo
        hi = up  # vuot tran: tim delta lon nhat trong [lo, hi] bang zip that
        while hi - lo > 1 and len(tried) < 16:
            mid = (lo + hi) // 2
            if fits(mid) and grew(mid, lo):
                lo = mid
            else:
                hi = mid
        break
    while lo + 1 not in tried and len(tried) < 16:  # buoc +1 cuoi: dam bao lo+1 khong duoi tran
        if fits(lo + 1) and grew(lo + 1, lo):
            lo += 1
        else:
            break
    return lo, [{"delta": k, "code": v["code"], "zip_bytes": v["zip_bytes"]}
                for k, v in sorted(tried.items())]


def assert_clean(sub: list, bad: set) -> dict:
    """0 bad id trong bai; moi chunk nam trong doc list; 1.200 cau id 1-1200."""
    bs = {str(i) for i in bad}
    assert len(sub) == 1200, len(sub)
    assert {int(it["id"]) for it in sub} == set(range(1, 1201))
    bad_docs = bad_chunks = ngoai_doclist = 0
    for it in sub:
        docs = {str(d) for d in it["relevant_docs"]}
        bad_docs += len(docs & bs)
        for c in it["relevant_chunks"]:
            bad_chunks += str(c["doc_id"]) in bs
            ngoai_doclist += str(c["doc_id"]) not in docs
    assert bad_docs == 0 and bad_chunks == 0 and ngoai_doclist == 0, \
        (bad_docs, bad_chunks, ngoai_doclist)
    return {"cau": len(sub), "bad_doc": bad_docs, "bad_chunk": bad_chunks,
            "chunk_ngoai_doclist": ngoai_doclist}


# ---------------------------------------------------------------- stage

def stage_verify() -> None:
    t0 = time.time()
    bad_list = read_ids_tsv(BAD_TSV)
    fp_list = read_ids_tsv(FP_TSV)
    bad, fp = set(bad_list), set(fp_list)
    assert len(bad_list) == len(bad), "bad_ids trung nhau"
    old, fix = {}, {}
    for tag in PKG_TAGS:
        old[tag] = scan_package(CRAWL / f"handoff_vi_{tag}")
        fix[tag] = scan_package(CRAWL / f"handoff_vi_{tag}_fix")
        print("SCAN", tag, "cu", old[tag]["records"], "fix", fix[tag]["records"], flush=True)
    import numpy as np
    rep = hypothesis_report(bad, fp,
                            {t: old[t]["ids"] for t in PKG_TAGS},
                            {t: fix[t]["ids"] for t in PKG_TAGS},
                            set(map(int, np.load(IDX_IDS))))
    rep["goi"] = {t: {"cu": {k: v for k, v in old[t].items() if k != "ids"},
                      "fix": {k: v for k, v in fix[t].items() if k != "ids"}}
                  for t in PKG_TAGS}
    sub = read(REF_SUB)
    rep["anh_huong_6757"] = impact_report(sub, bad, read(REF_C), read(REF_K))
    rep["giay"] = round(time.time() - t0, 1)
    write(OUT / "verify_report.json", rep)
    write(OUT / "bad_ids.json", sorted(bad))
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    if not rep["ok_thuyet"]:
        raise SystemExit(2)


def stage_sanity() -> None:
    write_inputs(set(), IN0)
    vi(*chain_args(SANITY_NAME, SANITY_DELTA, IN0))
    cmp_rep = core.compare_subs(read(OUT / f"submission_{SANITY_NAME}.json"), str(REF_SUB))
    write(OUT / "sanity_report.json", cmp_rep)
    print(json.dumps(cmp_rep, ensure_ascii=False), flush=True)
    require_sanity(cmp_rep)


def stage_build() -> None:
    if (OUT / "build_report_vi28.json").exists() and (OUT / f"submission_{NAME}.zip").exists():
        print("build da xong (resume): bo qua", flush=True)
        return
    bad = load_bad_ids()
    assert len(bad) == 16079, len(bad)
    assert not (bad & set(map(int, read_ids_tsv(FP_TSV)))), "fp phai ngoai tap bad"
    loc_rep = write_inputs(bad, IN1)
    delta, attempts = find_delta(IN1, MAX_ZIP, DELTA0, PROBE)
    r = chain_zip_check(NAME, delta, MAX_ZIP, IN1, validate=True)
    assert r["code"] == 0, r
    clean = assert_clean(read(OUT / f"submission_{NAME}.json"), bad)
    fp = fingerprint(OUT / f"submission_{NAME}.zip")
    write(OUT / "build_report_vi28.json",
          {"name": NAME, "delta": delta, "attempts": attempts, "max_zip": MAX_ZIP,
           "loc_dau_vao": loc_rep, "assert_clean": clean, "validate": "ok",
           "zip": fp, "n_q_TB": r.get("n_q_TB"), "tong_chunk": r.get("tong_chunk")})
    print(json.dumps({"build": NAME, "delta": delta, "attempts": attempts, **clean,
                      "zip_bytes": fp["bytes"], "sha256": fp["sha256"]}, ensure_ascii=False),
          flush=True)


def sub_row(sub: list, rep_chain: dict, fp: dict, delta) -> dict:
    n = max(len(sub), 1)
    docs_chunk = sum(len({str(c["doc_id"]) for c in it["relevant_chunks"]}) for it in sub)
    return {"so_cau": len(sub), "delta": delta,
            "zip_bytes": fp["bytes"], "sha256": fp["sha256"],
            "TB_doc_cau": round(sum(len(it["relevant_docs"]) for it in sub) / n, 2),
            "n_q_TB": (rep_chain or {}).get("n_q_TB"),
            "TB_chunk_cau": round(sum(len(it["relevant_chunks"]) for it in sub) / n, 2),
            "TB_doc_co_chunk_cau": round(docs_chunk / n, 2),
            "tong_chunk": sum(len(it["relevant_chunks"]) for it in sub)}


def stage_report() -> None:
    sub = read(OUT / f"submission_{NAME}.json")
    ref = read(REF_SUB)
    rep28 = read(OUT / f"report_chain_{NAME}.json")
    rep99 = read(W / "report_chain_vi25_z99.json")
    r28 = sub_row(sub, rep28, fingerprint(OUT / f"submission_{NAME}.zip"), rep28.get("delta"))
    r99 = sub_row(ref, rep99, fingerprint(REF_ZIP), rep99.get("delta"))
    dl28 = {str(it["id"]): [str(d) for d in it["relevant_docs"]] for it in sub}
    dl99 = {str(it["id"]): [str(d) for d in it["relevant_docs"]] for it in ref}
    doi = trung30 = 0
    for q, docs in dl28.items():
        other = dl99.get(q, [])
        doi += docs != other
        trung30 += len(set(docs[:30]) & set(other[:30]))
    n = max(len(sub), 1)
    r28.update({"so_cau_doc_thay_doi_vs_z99": doi,
                "TB_top30_doc_trung_z99": round(trung30 / n, 2)})
    br = read(OUT / "build_report_vi28.json") if (OUT / "build_report_vi28.json").exists() else {}
    rep = {"vi28": r28, "z99_tham_chieu": r99, "attempts": br.get("attempts", []),
           "assert_clean": br.get("assert_clean"), "validate": "ok"}
    write(OUT / "report_vi28.json", rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


def status() -> None:
    print("time", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    for p in sorted(OUT.glob("*.log")) + sorted(OUT.glob("*.json")):
        print(p.name, p.stat().st_size, time.ctime(p.stat().st_mtime), flush=True)
    print("locks", {p.name: p.read_text(errors="replace")
                    for p in (ROOT / "locks").glob("*.lock")}, flush=True)


def cleanup() -> None:
    lock = ROOT / "locks" / "heavy_job.lock"
    if lock.exists() and "vi28" in lock.read_text(errors="replace"):
        raise RuntimeError("vi28 lock still exists; inspect job before cleanup")
    import subprocess
    name = "r2ai_vi28_run"
    q = subprocess.run(["schtasks", "/Query", "/TN", name, "/FO", "CSV", "/NH"],
                       capture_output=True, text=True)
    if not q.returncode:
        assert '"Running"' not in q.stdout, name + " still running"
        subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], check=True)
    print("CLEANUP DONE", "locks", {p.name: p.read_text(errors="replace")
                                    for p in (ROOT / "locks").glob("*.lock")}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["verify", "sanity", "build", "report", "status", "cleanup"])
    args = parser.parse_args()
    if args.stage == "status":
        status()
        return
    if args.stage == "cleanup":
        cleanup()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with owned_lock(ROOT / "locks" / "heavy_job.lock", f"vi28 {args.stage} pid={os.getpid()}"):
        globals()["stage_" + args.stage]()
    print("DONE", args.stage, flush=True)


if __name__ == "__main__":
    main()
