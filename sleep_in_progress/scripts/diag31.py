r"""diag31 = docceil tren pool vi30: do tran Docs Recall cua pool (chuan doan, mot bai nop).

docceil  relevant_docs = TOAN BO pool_v30 (da loc id bad + ngoai kho) theo thu tu fhyde
         truoc + pool sau (diag13.order_pool_docs); noi them CUOI danh sach cac doc cua
         submission_vi30 neu thieu (giu thu tu), roi doc cua chunk (diag13.with_chunk_docs).
         relevant_chunks = chunk vi30 y nguyen. Ghi json + zip LZMA (zip_submission.py).
         Neu zip > 99.000.000 B: cat chunk CUOI moi cau theo ti le deu (nhi phan so chunk
         moi cau giu lai) den khi <= tran. Ghi so chunk da cat.
         assert_clean (vi29) voi tap loai vi30 (bad chua crawl lai U ngoai kho) + kho chuan.

Dung:
  python scripts/diag31.py docceil --sub SUB30.json --pool POOL.json --fhyde FHYDE.json
      --out submission_docceil30.json --zip submission_docceil30.zip --report docceil_report.json
"""
import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diag13 import dist, order_pool_docs, with_chunk_docs

MAX_ZIP = 99_000_000


def build_docceil30(sub30: list, pool: dict, fhyde: dict) -> tuple[list, list, list]:
    """relevant_docs = toan bo pool (fhyde truoc, pool sau) + doc vi30 thieu noi cuoi +
    doc chunk thieu; chunk y nguyen vi30. Tra (out, doc_vi30_them, doc_chunk_them)."""
    po = {str(k): v for k, v in pool.items()}
    fu = {str(k): v for k, v in fhyde.items()}
    out, add_vi30, add_chunk = [], [], []
    for it in sub30:
        q = str(it["id"])
        docs = order_pool_docs(po.get(q, []), fu.get(q, []))
        seen = {str(d) for d in docs}
        a0 = 0
        for d in it["relevant_docs"]:
            if str(d) not in seen:
                seen.add(str(d))
                docs.append(d)
                a0 += 1
        docs, a1 = with_chunk_docs(docs, it["relevant_chunks"])
        add_vi30.append(a0)
        add_chunk.append(a1)
        out.append({"id": it["id"], "relevant_docs": docs,
                    "relevant_chunks": it["relevant_chunks"]})
    return out, add_vi30, add_chunk


def cut_chunks(sub: list, k: int) -> list:
    """Giu k chunk dau moi cau (cat CUOI danh sach); giu nguyen docs + id."""
    return [{"id": it["id"], "relevant_docs": it["relevant_docs"],
             "relevant_chunks": it["relevant_chunks"][:k]} for it in sub]


def zip_lzma(json_path, zip_path) -> int:
    """Zip 1 entry submission.json bang ZIP_LZMA (giong zip_submission.pack --method lzma)."""
    with zipfile.ZipFile(str(zip_path), "w", zipfile.ZIP_LZMA) as z:
        z.write(str(json_path), arcname="submission.json")
    return Path(zip_path).stat().st_size


def write_json(sub: list, path) -> int:
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(sub, f, ensure_ascii=False)
    return Path(path).stat().st_size


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def cmd_docceil(a) -> None:
    t0 = time.time()
    import vi29
    import vi30
    from vi24 import read
    from zip_submission import pack

    sub30 = json.loads(Path(a.sub).read_text(encoding="utf-8"))
    pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))
    fhyde = json.loads(Path(a.fhyde).read_text(encoding="utf-8"))
    out, add_vi30, add_chunk = build_docceil30(sub30, pool, fhyde)
    n = max(len(out), 1)
    docs_cu = [len(it["relevant_docs"]) for it in sub30]
    docs_moi = [len(it["relevant_docs"]) for it in out]
    nc_full = [len(it["relevant_chunks"]) for it in out]
    del sub30, pool, fhyde

    out_path, zip_path = Path(a.out), Path(a.zip)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_nc = max(nc_full, default=0)
    k = None
    write_json(out, out_path)
    size = zip_lzma(out_path, zip_path)
    print(f"FULL k={max_nc} json={out_path.stat().st_size} zip={size}", flush=True)
    if size > MAX_ZIP and max_nc > 0:
        lo, hi, best = 0, max_nc - 1, 0
        while lo <= hi:
            mid = (lo + hi) // 2
            write_json(cut_chunks(out, mid), out_path)
            s = zip_lzma(out_path, zip_path)
            print(f"PROBE k={mid} zip={s}", flush=True)
            if s <= MAX_ZIP:
                best, lo = mid, mid + 1
            else:
                hi = mid - 1
        k = best
        write_json(cut_chunks(out, k), out_path)
    if k is not None:
        out = cut_chunks(out, k)
    assert out_path.stat().st_size > 0
    st = pack(str(out_path), str(zip_path), "lzma")
    size = st["zip_bytes"]
    assert size <= MAX_ZIP, size

    bad = vi30.load_bad()
    added = read(vi30.OUT / "new_ids.json")
    excl = vi30.build_excl(bad, added)
    corpus = vi29.load_corpus_ids(vi30.LINKS)
    clean = vi29.assert_clean(out, excl, corpus)

    kept = [len(it["relevant_chunks"]) for it in out]
    fp = {"bytes": size, "sha256": sha256_file(zip_path)}
    rep = {"name": "docceil30", "cau": len(out),
           "doc_moi_cau": dist(docs_moi), "doc_vi30_cau": dist(docs_cu),
           "TB_doc_them_vs_vi30": round(sum(docs_moi) / n - sum(docs_cu) / n, 2),
           "tong_doc_them_vs_vi30": sum(docs_moi) - sum(docs_cu),
           "doc_vi30_thieu_pool_noi_them": {"tong": sum(add_vi30), "TB": sum(add_vi30) / n},
           "doc_chunk_them": {"tong": sum(add_chunk), "TB": sum(add_chunk) / n},
           "chunk_giu_moi_cau": dist(kept),
           "k_giu": k if k is not None else max_nc,
           "chunk_cat": sum(nc_full) - sum(kept),
           "tong_chunk_goc": sum(nc_full), "tong_chunk_giu": sum(kept),
           "zip_limit": MAX_ZIP, "zip_bytes": fp["bytes"], "json_bytes": st["json_bytes"],
           "sha256": fp["sha256"], "assert_clean": clean, "validate": "bat",
           "giay": round(time.time() - t0, 1)}
    if a.report:
        Path(a.report).write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    print(f"ghi {out_path} + {zip_path}: {len(out)} cau, doc/cau TB {sum(docs_moi) / n:.2f} "
          f"min {min(docs_moi)} max {max(docs_moi)}, zip {fp['bytes']} B, sha256 {fp['sha256']}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    dc = sp.add_parser("docceil")
    dc.add_argument("--sub", required=True, help="submission_vi30.json")
    dc.add_argument("--pool", required=True, help="in1/pool_v30.json (da loc)")
    dc.add_argument("--fhyde", required=True, help="cache_fhyde_v30.json (chua loc)")
    dc.add_argument("--out", required=True)
    dc.add_argument("--zip", required=True)
    dc.add_argument("--report", default="")
    a = ap.parse_args()
    {"docceil": cmd_docceil}[a.cmd](a)


if __name__ == "__main__":
    main()
