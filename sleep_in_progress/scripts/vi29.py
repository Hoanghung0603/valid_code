r"""vi29 = vi28 + 2 sua loi (mot bai nop).

Loi A (da xac minh): diag14.order_q bo trung theo chunk_text (r[4]) TOAN cau, khong xet
doc_id, va cx93.filter_base chay SAU order_q (vi23.py:515). He qua: chunk cua doc trong
doclist co the bi loai vi mot ban cung text thuoc doc khac (thuong la doc ngoai doclist,
hoac trung trang khac id). Luat cham khop chunk theo cung doc_id nen ban do la chunk hop
le rieng. Sua: bo trung theo cap (str(doc_id), chunk_text), ap dung sau/ket hop filter_base
(tuc: loc doclist truoc, roi bo trung (doc_id, text)). zc van tinh tren MOI dong ung vien
cua cau (giu quy uoc), zd, s, sap xep, apply_bonus b_cov 1, kprime giu nguyen. Khong sua
diag14.py/cx93.py/vi23.py (cac bai cu phai tai lap duoc): order_q_docdedup +
chain_build_v29 voi co dedup_mode in {"text", "doc_text"} ("text" = pipeline cu).

Loi B: 9.298 id trong chi muc (613.087 doc kho cu) khong thuoc kho chuan moi
links_corpus_vi_en_zh.parquet (cot id, b660: D:\r2ai\data\sub\diag19\). Nhan vang chi
co id trong kho chuan => cac id nay chac chan sai. Sua: loc moi id khong thuoc kho chuan
khoi MOI dau vao chain (pool, q8b, dense, hyde, cands, ref_chunks), cung cach vi28.write_inputs
(giu thu tu, khong lap cho). Loc chong len tap bad 16.079 cua vi28 (tap loai = bad +
ngoai kho). k_q van lay tu REF_K (diag14/submission_c120in.json) nhu vi28, khong doi.

Stage (deploy/b660_vi29.bat; schtasks r2ai_vi29_run; D:\r2ai\data\sub\diag29\):
  count      do luong tren chinh vi28 dau vao in1 + submission_vi28.json: so id ngoai kho
             trong chi muc/pool; so cap (q,doc) ngoai kho trong relevant_docs cua vi28
             (TB/cau, so cau); so chunk cua chung trong vi28; so id ngoai kho trong c120in
             va cf4. Loi A: tong so chunk ung vien thuoc doclist vi28 bi order_q loai do
             trung text voi doc khac; so cau anh huong; trong do bao nhieu chunk se lot
             top-k' (so k' moi) -> count_report.json
  sanity     chain_build_v29 dedup_mode="text", tap loai = bad_vi28 (khong loc ngoai kho),
             delta 57 -> phai tai lap submission_vi28.json 1200/1200 (doc + chunk, dung
             thu tu). Lech -> exit 2, dung.
  build      dedup_mode="doc_text" + tap loai = bad + ngoai kho; find_delta nhu vi28 (zip
             LZMA THAT <= 99.000.000 B, khoi dau 57); evaluate validate;
             zip_submission --method lzma; check_zip; assert 1200 cau id 1..1200, 0 id
             bad/ngoi kho trong docs va chunks, moi chunk nam trong doclist, khong co cap
             (doc_id,text) trung trong mot cau.
  report     so vi29 voi vi28: TB doc/cau, n_q TB, TB chunk/cau, doc co chunk/cau, delta,
             byte zip, sha256, so cau doclist doi, top-30 trung, so chunk moi.
  tokmeasure CHI DO (khong chan stage 1-4; chay sau cung): do muc cat token cua docwin
             (cua so 300 am tiet buoc 250 + query, --max-len 512 trong docwin_run.py) tren
             mau ~300 cau x ~10 doc (seed co dinh), tai tao cua o dung docwin_run.
             Tokenizer bge-reranker-v2-m3 (XLM-R): transformers/tokenizers + file tokenizer
             neu co; khong thi chi tai tokenizer.json (khong weights) + goi `tokenizers`.
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vi23 as core
import vi28 as v28
from vi24 import fingerprint, owned_lock, read, run, vi, write

ROOT = Path(os.environ.get("R2AI_ROOT", "D:/r2ai"))
W = ROOT / "data/sub/diag25"                    # vi25 - CHI DOC
OUT28 = ROOT / "data/sub/diag28"                # vi28 - CHI DOC
IN1_28 = OUT28 / "in1"                          # dau vao vi28 (bad da loc) - CHI DOC
OUT = ROOT / "data/sub/diag29"
IN2 = OUT / "in1"                               # dau vao vi29 (bad + ngoai kho da loc)
CORPUS = ROOT / "data/sub/diag19/links_corpus_vi_en_zh.parquet"
IDX_IDS = ROOT / "data/index/vi25/ids.npy"
Q8 = ROOT / "data/sub/qwen8b/cache_qwen8b_sorted.json"
REF_K = ROOT / "data/sub/diag14/submission_c120in.json"
REF_C = ROOT / "data/sub/diag11/chunks_rrf3d2_cf4.json"
SUB28 = OUT28 / "submission_vi28.json"
QUERY_PARQUET = ROOT / "data/query.parquet"
DOCS_TXT = [W / "docs_vi25_docwin.jsonl.zst",
            ROOT / "data/sub/chunkdocs100/chunkdocs100.part000.jsonl.zst",
            ROOT / "data/sub/chunkdocs100/chunkdocs100.part001.jsonl.zst",
            ROOT / "data/sub/docwin_ext/docwin_ext.part000.jsonl.zst",
            ROOT / "data/sub/docwin_ext/docwin_ext.part001.jsonl.zst",
            ROOT / "data/sub/chunkdocs50/chunkdocs50.part000.jsonl.zst",
            ROOT / "data/sub/chunkdocs/chunkdocs.part000.jsonl.zst",
            W / "docs_vi25_embed.jsonl.zst", W / "docs_vi25_old.part000.jsonl.zst"]
TOK_DIR = OUT / "tok"
TOK_JSON = TOK_DIR / "bge-reranker-v2-m3_tokenizer.json"
TOK_URL = "https://huggingface.co/BAAI/bge-reranker-v2-m3/resolve/main/tokenizer.json"
TOK_MODEL = "BAAI/bge-reranker-v2-m3"
TOKENIZERS_PIN = "tokenizers==0.21.4"
NAME = "vi29"
SANITY_NAME = "vi29_sanity"
PROBE = "vi29p"
SANITY_DELTA = 57
DELTA0 = 57
MAX_ZIP = 99_000_000
THR, LO, HI, WD, B_COV = -2.0, 40, 400, 4.0, 1.0
FILE_NAMES28 = {"pool": "pool_v28.json", "q8b": "q8b_v28.json", "dense": "dense_v28.json",
                "hyde": "hyde_v28.json", "cands": "cands_v28.json.zst",
                "ref_chunks": "ref_chunks_v28.json"}
FILE_NAMES = {"pool": "pool_v29.json", "q8b": "q8b_v29.json", "dense": "dense_v29.json",
              "hyde": "hyde_v29.json", "cands": "cands_v29.json.zst",
              "ref_chunks": "ref_chunks_v29.json"}
WINDOW_WORDS, WINDOW_STRIDE, MAX_LEN = 300, 250, 512
TOK_SEED, TOK_N_Q, TOK_N_DOC = 2929, 300, 10


# ---------------------------------------------------------------- tien ich loc

def load_corpus_ids(path=None) -> set:
    """Tap id (int) cua kho chuan moi links_corpus_vi_en_zh.parquet, cot id."""
    import pyarrow.parquet as pq
    t = pq.read_table(str(path or CORPUS), columns=["id"])
    return {int(i) for i in t.column("id").to_pylist()}


def _strset(ids) -> set:
    return {str(i) for i in ids}


def _keep(doc_id, bad_s: set, corpus_s) -> bool:
    s = str(doc_id)
    return s not in bad_s and (corpus_s is None or s in corpus_s)


def filter_cache(cache: dict, bad: set, corpus=None) -> dict:
    """{qid: hits} sau khi bo hit bad + ngoai kho (giu key + thu tu, khong lap cho)."""
    bs, cs = _strset(bad), _strset(corpus) if corpus is not None else None
    return {str(q): [h for h in hits or [] if _keep(h[0], bs, cs)] for q, hits in cache.items()}


def filter_cands(cands: dict, bad: set, corpus=None) -> dict:
    """{qid: [doc_id, doc_rank, chunk_idx, logit, chunk_text]} bo dong chunk bad + ngoai kho."""
    bs, cs = _strset(bad), _strset(corpus) if corpus is not None else None
    return {str(q): [r for r in rows if _keep(r[0], bs, cs)] for q, rows in cands.items()}


def filter_ref_chunks(ref: list, bad: set, corpus=None) -> list:
    """ref_chunks bo chunk cua doc bad + ngoai kho; giu cau + thu tu."""
    bs, cs = _strset(bad), _strset(corpus) if corpus is not None else None
    out = []
    for it in ref or []:
        kept = [c for c in it.get("relevant_chunks", []) if _keep(c["doc_id"], bs, cs)]
        out.append({**it, "relevant_chunks": kept})
    return out


def write_inputs(bad: set, indir: Path, corpus=None) -> dict:
    """Ghi 6 dau vao chain da loc (bad + ngoai kho) vao indir; tra so lieu da gỡ.

    Giong vi28.write_inputs (giu thu tu, khong lap cho); them dieu kien id thuoc kho chuan
    khi corpus khac None. Nguon doc tu W/Q8/REF_C (CHI DOC)."""
    indir.mkdir(parents=True, exist_ok=True)
    spec = [(FILE_NAMES["pool"], W / "cache_pool_v25.json", "cache"),
            (FILE_NAMES["q8b"], Q8, "cache"),
            (FILE_NAMES["dense"], W / "dense_v25_top200.json", "cache"),
            (FILE_NAMES["hyde"], W / "hyde_v25_top200.json", "cache"),
            (FILE_NAMES["cands"], W / "chunk_dw_u900_v25.json.zst", "cands"),
            (FILE_NAMES["ref_chunks"], REF_C, "ref")]
    rep = {"bad_id": len(bad), "co_loc_ngoai_kho": corpus is not None,
           "out_dir": str(indir), "files": {}}
    for name, src, kind in spec:
        obj = read(src)
        if kind == "cache":
            out_obj = filter_cache(obj, bad, corpus)
            n_before = sum(len(v) for v in obj.values())
            n_after = sum(len(v) for v in out_obj.values())
        elif kind == "cands":
            out_obj = filter_cands(obj, bad, corpus)
            n_before = sum(len(v) for v in obj.values())
            n_after = sum(len(v) for v in out_obj.values())
        else:
            out_obj = filter_ref_chunks(obj, bad, corpus)
            n_before = sum(len(it.get("relevant_chunks", [])) for it in obj)
            n_after = sum(len(it["relevant_chunks"]) for it in out_obj)
        dest = indir / name
        if name.endswith(".zst"):
            nbytes = v28.write_zst(dest, out_obj)
        else:
            nbytes = write(dest, out_obj)
        rep["files"][name] = {"nguon": str(src), "truoc": n_before, "sau": n_after,
                              "bo": n_before - n_after, "json_bytes": nbytes}
        del obj, out_obj
    return rep


# ---------------------------------------------------------------- loi A: order_q moi

def order_q_docdedup(rows: list, fused_hits: list, allowed=None, dedup_mode: str = "doc_text",
                     wd: float = WD) -> list[dict]:
    """Giong diag14.order_q nhung dedup_mode="doc_text": loc doclist TRUOC, roi bo trung theo
    cap (str(doc_id), chunk_text), giu dong truoc theo s giam dan.

    dedup_mode="text": tai lap dung order_q + filter_base cu (bo trung chunk_text toan cau
    roi loc doclist) de cac bai cu tai lap duoc. Moi phan tu: doc_id, chunk_text, logit, zc,
    zd, s. zc luon tinh tren MOI dong ung vien cua cau (quy uoc giu nguyen)."""
    if dedup_mode not in ("text", "doc_text"):
        raise ValueError(f"dedup_mode la: {dedup_mode}")
    rows = rows or []
    if dedup_mode == "text":
        from cx93 import filter_base
        from diag14 import order_q
        out = order_q(rows, fused_hits or [], wd)
        return filter_base(out, allowed) if allowed is not None else out
    from chunk_docfuse import doc_z, zscores
    zc = zscores([float(r[3]) for r in rows])
    zdm = doc_z(fused_hits or [])
    zmin = min(zdm.values()) if zdm else 0.0
    scored = [(c + wd * zdm.get(str(r[0]), zmin), c, zdm.get(str(r[0]), zmin), r)
              for c, r in zip(zc, rows)]
    order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
    out, seen = [], set()
    for i in order:
        s, c, z, r = scored[i]
        if allowed is not None and str(r[0]) not in allowed:
            continue
        key = (str(r[0]), r[4])
        if key in seen:
            continue
        seen.add(key)
        out.append({"doc_id": r[0], "chunk_text": r[4], "logit": float(r[3]),
                    "zc": c, "zd": z, "s": s})
    return out


def chain_build_v29(pool: dict, q8b: dict, dense: dict, hyde: dict, cands: dict, kq: dict,
                    ref_chunks: dict, thr: float = THR, lo: int = LO, hi: int = HI,
                    wd: float = WD, b_cov: float = B_COV, delta: int = DELTA0,
                    dedup_mode: str = "text", doclists=None) -> tuple:
    """Sao chep vi23.chain_build voi co dedup_mode: "text" = pipeline cu (order_q +
    filter_base), "doc_text" = loc doclist truoc + bo trung (doc_id, text). Con lai giu
    nguyen: fuse -> doclist -> chon chunk (k' = min(k_q+delta, |C_q|), apply_bonus b_cov)."""
    if dedup_mode not in ("text", "doc_text"):
        raise ValueError(f"dedup_mode la: {dedup_mode}")
    from cx110 import hist_from_pairs
    from cx130 import kprime
    from cx93 import apply_bonus
    from diag14 import strip_meta
    from fuse3 import fuse3 as fuse3_fn
    from union_sub import adaptive_n

    rrf3d2 = fuse3_fn(pool, q8b, dense, (core.W_BGE, core.W_Q8B, core.W_DENSE, 0.0),
                      core.K_RRF, None, None)
    # vi33: trong so 8B rieng cho fhyde (thu tu chon doc); khong dat thi = W_Q8B (bai cu y nguyen).
    # rrf3d2 (nguon zd cua chunk) luon dung W_Q8B.
    w8f = getattr(core, "W_Q8B_FHYDE", None)
    w8f = core.W_Q8B if w8f is None else w8f
    fhyde = fuse3_fn(pool, q8b, dense, (core.W_BGE, w8f, core.W_DENSE, core.W_HYDE),
                     core.K_RRF, hyde, None)
    # vi35: nqf = (method, thr) -> n_q = pick_nq_fused(fhyde[q]); None -> adaptive_n bge (bai cu).
    nqf = getattr(core, "NQ_FUSED", None)
    if nqf is not None:
        from diag13 import pick_nq_fused
    sub, ns, kps, caps, pairs_hist = [], [], [], [], []
    for q in sorted(pool, key=lambda x: int(x)):
        if nqf is None:
            n = adaptive_n(pool[q], thr, lo, hi)
        else:
            n = pick_nq_fused(fhyde.get(q, []), nqf[1], lo, hi, nqf[0])
        base = core.first_n_docs(fhyde.get(q, []), n)
        docs = core.full8b_docs(base, ref_chunks.get(q, []))
        if doclists is not None:
            docs = list(doclists[q])
            n = len(docs)
        allowed = {str(d) for d in docs}
        cin = order_q_docdedup(cands.get(q, []), rrf3d2.get(q, []), allowed, dedup_mode, wd)
        kp = kprime(kq.get(q, 0), len(cin), delta)
        sel = strip_meta(apply_bonus(cin, b_cov), kp)
        for c in sel:
            assert str(c["doc_id"]) in allowed
        sub.append({"id": int(q) if str(q).isdigit() else q,
                    "relevant_docs": docs, "relevant_chunks": sel})
        ns.append(n)
        kps.append(kp)
        caps.append(len(cin))
        pairs_hist.append((kq.get(q, 0), len(cin)))
    n_q = max(len(sub), 1)
    rep = {"n_q_TB": sum(ns) / n_q, "n_q_min": min(ns), "n_q_max": max(ns),
           "TB_doc_cau": sum(len(s["relevant_docs"]) for s in sub) / n_q,
           "TB_chunk_cau": sum(kps) / n_q, "tong_chunk": sum(kps),
           "min_chunk": min(kps), "max_chunk": max(kps),
           "TB_C_q": sum(caps) / n_q, "cau_chan_boi_C_q": sum(1 for k, c in zip(kps, caps) if c < k),
           "delta": delta, "thr": thr, "lo": lo, "hi": hi, "wd": wd, "b_cov": b_cov,
           "dedup_mode": dedup_mode,
           "nq_mode": "bge" if nqf is None else f"fused_{nqf[0]}",
           "nq_fused_thr": None if nqf is None else nqf[1],
           "hist_kcap": hist_from_pairs(pairs_hist)}
    return rrf3d2, fhyde, sub, rep


# ---------------------------------------------------------------- chain runner

def chain_files(indir: Path, names: dict) -> dict:
    """Duong dan 6 dau vao chain + ref-kq (k_q tu REF_K, khong loc)."""
    return {"pool": indir / names["pool"], "q8b": indir / names["q8b"],
            "dense": indir / names["dense"], "hyde": indir / names["hyde"],
            "cands": indir / names["cands"], "ref_chunks": indir / names["ref_chunks"],
            "ref_kq": REF_K}


def run_chain(name: str, delta: int, files: dict, dedup_mode: str, out_dir: Path,
              write_caches: bool = False) -> dict:
    """Chay chain_build_v29 va ghi submission_<name>.json + report_chain_<name>.json
    (giong vi23.cmd_chain de check_zip dung duoc)."""
    t0 = time.time()
    pool = core.load_cache(files["pool"])
    q8b = core.load_cache(files["q8b"])
    dense = core.load_cache(files["dense"])
    hyde = core.load_cache(files["hyde"]) if files.get("hyde") else {}
    cands = {str(q): v for q, v in core.load_json(files["cands"]).items()}
    ref = core.load_json(files["ref_chunks"]) if files.get("ref_chunks") else []
    kq = {str(it["id"]): len(it["relevant_chunks"]) for it in core.load_json(files["ref_kq"])}
    ref_map = {str(it["id"]): it["relevant_chunks"] for it in ref} if ref else {
        str(it["id"]): it["relevant_chunks"] for it in core.load_json(files["ref_kq"])}
    rrf3d2, fhyde, sub, rep = chain_build_v29(pool, q8b, dense, hyde, cands, kq, ref_map,
                                              THR, LO, HI, WD, B_COV, delta, dedup_mode)
    if write_caches:
        core.write_json(out_dir / f"cache_rrf3d2_{dedup_mode}.json", rrf3d2)
        core.write_json(out_dir / f"cache_fhyde_{dedup_mode}.json", fhyde)
    fp = out_dir / f"submission_{name}.json"
    core.write_json(fp, sub)
    core.write_json(out_dir / f"doclist_{name}.json",
                    {str(it["id"]): it["relevant_docs"] for it in sub})
    rep.update({"mode": dedup_mode, "name": name, "json_bytes": fp.stat().st_size,
                "cands": str(files["cands"]), "pool": str(files["pool"]),
                "thoi_gian_giay": round(time.time() - t0, 1)})
    core.write_json(out_dir / f"report_chain_{name}.json", rep)
    print(json.dumps({k: v for k, v in rep.items() if k != "hist_kcap"}, ensure_ascii=False),
          flush=True)
    return rep


def chain_zip_check(name: str, delta: int, max_zip: int, files: dict, dedup_mode: str,
                    validate: bool = False) -> dict:
    """Chain voi delta, (tuy chon) evaluate validate, zip LZMA that, check_zip. Mot lan thu."""
    run_chain(name, delta, files, dedup_mode, OUT)
    if validate:
        run("evaluate.py", "validate", OUT / f"submission_{name}.json")
    run("zip_submission.py", "--json", OUT / f"submission_{name}.json",
        "--zip", OUT / f"submission_{name}.zip", "--method", "lzma")
    code = core.check_zip(str(OUT / f"submission_{name}.json"),
                          str(OUT / f"submission_{name}.zip"),
                          str(OUT / f"report_chain_{name}.json"), name, max_zip=max_zip)
    rep = read(OUT / f"report_chain_{name}.json")
    entry = rep.get("zip", {}).get(name, {})
    return {"code": code, "next_delta": entry.get("next_delta"), "zip_bytes": entry.get("zip_bytes"),
            "json_bytes": rep.get("json_bytes"), "tong_chunk": rep.get("tong_chunk"),
            "hist_kcap": rep.get("hist_kcap"), "n_q_TB": rep.get("n_q_TB")}


def find_delta(files: dict, dedup_mode: str, max_zip: int, delta0: int, probe: str) -> tuple:
    """Giong vi28.find_delta: delta LON NHAT voi zip LZMA THAT <= max_zip; cache theo lan thu.

    Giam bang vong check-zip/next_delta den khi duoi tran, roi uoc luong cx130.largest_delta_within
    + nhi phan/+1 xac nhan bang zip that. Tong chunk khong tang nua = bao hoa |C_q| -> dung.
    Ket qua moi delta duoc cache de resume."""
    from cx130 import largest_delta_within
    cache_file = OUT / f"attempts_{probe}.json"
    tried = {int(k): v for k, v in (read(cache_file) if cache_file.exists() else {}).items()}

    def attempt(delta):
        if delta not in tried:
            tried[delta] = chain_zip_check(probe, delta, max_zip, files, dedup_mode)
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


def require_sanity(cmp_rep: dict) -> None:
    """Sanity phai khop 1200/1200 ca doc lan chunk, khong thi dung (exit 2)."""
    ok = (cmp_rep.get("ok") and cmp_rep.get("so_cau") == 1200
          and cmp_rep.get("cau_khop_doc") == 1200
          and cmp_rep.get("cau_khop_chunk") == 1200)
    if not ok:
        print("SANITY FAIL", json.dumps(cmp_rep, ensure_ascii=False), flush=True)
        raise SystemExit(2)


def assert_clean(sub: list, bad: set, corpus=None) -> dict:
    """1.200 cau id 1-1200; 0 bad/ngoi kho trong docs va chunks; moi chunk trong doclist;
    khong co cap (doc_id, chunk_text) trung trong mot cau."""
    bs, cs = _strset(bad), _strset(corpus) if corpus is not None else None
    assert len(sub) == 1200, len(sub)
    assert {int(it["id"]) for it in sub} == set(range(1, 1201))
    bad_docs = bad_chunks = ngoai_doclist = ngoai_kho_docs = ngoai_kho_chunks = trung_cap = 0
    for it in sub:
        docs = {str(d) for d in it["relevant_docs"]}
        bad_docs += len(docs & bs)
        if cs is not None:
            ngoai_kho_docs += len(docs - cs)
        seen = set()
        for c in it["relevant_chunks"]:
            d = str(c["doc_id"])
            bad_chunks += d in bs
            ngoai_doclist += d not in docs
            if cs is not None:
                ngoai_kho_chunks += d not in cs
            pair = (d, c["chunk_text"])
            trung_cap += pair in seen
            seen.add(pair)
    assert bad_docs == 0 and bad_chunks == 0 and ngoai_doclist == 0, \
        (bad_docs, bad_chunks, ngoai_doclist)
    assert ngoai_kho_docs == 0 and ngoai_kho_chunks == 0, (ngoai_kho_docs, ngoai_kho_chunks)
    assert trung_cap == 0, trung_cap
    return {"cau": len(sub), "bad_doc": bad_docs, "bad_chunk": bad_chunks,
            "chunk_ngoai_doclist": ngoai_doclist, "doc_ngoai_kho": ngoai_kho_docs,
            "chunk_ngoai_kho": ngoai_kho_chunks, "cap_doc_text_trung": trung_cap}


# ---------------------------------------------------------------- stage count

def ids_cache(cache: dict) -> set:
    return {str(h[0]) for hits in cache.values() for h in hits or []}


def ids_ref(items: list) -> set:
    out = set()
    for it in items or []:
        out |= {str(d) for d in it.get("relevant_docs", [])}
        out |= {str(c["doc_id"]) for c in it.get("relevant_chunks", [])}
    return out


def loi_a_report(cands: dict, doclists: dict, kq: dict, rrf3d2: dict,
                 wd: float = WD, delta: int = DELTA0) -> dict:
    """Anh huong loi A: dong ung vien thuoc doclist bi order_q loai sai vi trung text voi
    doc khac (ban giong text khac doc_id thang the). Dem dong bi loai sai, cap (doc_id,text)
    rieng, cau anh huong, va bao nhieu cap se lot top-k' voi k' = min(k_q+delta, |C_q|)
    theo luat vi29 (dedup doc_text)."""
    from chunk_docfuse import doc_z, zscores
    from cx93 import apply_bonus, filter_base
    from diag14 import order_q, strip_meta
    tong_dong = tong_cap = cau_anh = 0
    tong_vao = 0
    kps_cu, kps_moi, caps_cu, caps_moi = [], [], [], []
    for q in sorted(cands, key=lambda x: int(x)):
        rows = cands.get(q, []) or []
        allowed = {str(d) for d in doclists.get(q, [])}
        if not rows:
            continue
        zc = zscores([float(r[3]) for r in rows])
        zdm = doc_z(rrf3d2.get(q, []) or [])
        zmin = min(zdm.values()) if zdm else 0.0
        scored = [(c + wd * zdm.get(str(r[0]), zmin), c, zdm.get(str(r[0]), zmin), r)
                  for c, r in zip(zc, rows)]
        order = sorted(range(len(scored)), key=lambda i: -scored[i][0])
        win_doc, wrong = {}, []
        for i in order:
            r = scored[i][3]
            t, d = r[4], str(r[0])
            if t not in win_doc:
                win_doc[t] = d
            elif win_doc[t] != d and d in allowed:
                wrong.append(r)
        cin_old = filter_base(order_q(rows, rrf3d2.get(q, []) or [], wd), allowed)
        cin_new = order_q_docdedup(rows, rrf3d2.get(q, []) or [], allowed, "doc_text", wd)
        k_old = min(kq.get(q, 0) + delta, len(cin_old))
        k_new = min(kq.get(q, 0) + delta, len(cin_new))
        caps_cu.append(len(cin_old))
        caps_moi.append(len(cin_new))
        kps_cu.append(k_old)
        kps_moi.append(k_new)
        if not wrong:
            continue
        wrong_pairs = {(str(r[0]), r[4]) for r in wrong}
        sel = strip_meta(apply_bonus(cin_new, B_COV), k_new)
        sel_pairs = {(str(c["doc_id"]), c["chunk_text"]) for c in sel}
        vao = wrong_pairs & sel_pairs
        tong_dong += len(wrong)
        tong_cap += len(wrong_pairs)
        tong_vao += len(vao)
        cau_anh += 1
    n = max(len(cands), 1)
    return {"so_dong_bi_loai_sai": tong_dong, "so_cap_doc_text_bi_loai_sai": tong_cap,
            "so_cau_anh_huong": cau_anh, "so_cap_vao_topk_delta_%d" % delta: tong_vao,
            "delta_su_dung": delta,
            "TB_C_q_cu": round(sum(caps_cu) / max(len(caps_cu), 1), 2),
            "TB_C_q_moi": round(sum(caps_moi) / max(len(caps_moi), 1), 2),
            "TB_kprime_cu": round(sum(kps_cu) / max(len(kps_cu), 1), 2),
            "TB_kprime_moi": round(sum(kps_moi) / max(len(kps_moi), 1), 2),
            "so_cau": n}


def stage_count() -> None:
    t0 = time.time()
    corpus = load_corpus_ids()
    cs = _strset(corpus)
    bad = v28.load_bad_ids()
    import numpy as np
    idx_ids = [int(x) for x in np.load(IDX_IDS)]
    sub = read(SUB28)
    pool = core.load_cache(IN1_28 / FILE_NAMES28["pool"])
    cands = {str(q): v for q, v in core.load_json(IN1_28 / FILE_NAMES28["cands"]).items()}
    c120in = read(REF_K)
    cf4 = read(REF_C)

    pool_ids = ids_cache(pool)
    idx_ngoai = sum(1 for i in idx_ids if str(i) not in cs)
    pool_ngoai = sum(1 for i in pool_ids if i not in cs)
    cap_bad = chunk_bad = cau_anh = 0
    for it in sub:
        hit = sum(1 for d in {str(x) for x in it["relevant_docs"]} if d not in cs)
        cap_bad += hit
        cau_anh += bool(hit)
        chunk_bad += sum(1 for c in it["relevant_chunks"] if str(c["doc_id"]) not in cs)
    n = max(len(sub), 1)

    def ref_ngoai(items):
        ids = ids_ref(items)
        return {"so_id": len(ids), "so_id_ngoai_kho": sum(1 for i in ids if i not in cs)}

    rrf_path = OUT28 / "cache_rrf3d2_that.json"
    if rrf_path.exists():
        rrf3d2 = read(rrf_path)
    else:
        from fuse3 import fuse3 as fuse3_fn
        rrf3d2 = fuse3_fn(pool, core.load_cache(IN1_28 / FILE_NAMES28["q8b"]),
                          core.load_cache(IN1_28 / FILE_NAMES28["dense"]),
                          (core.W_BGE, core.W_Q8B, core.W_DENSE, 0.0), core.K_RRF, None, None)
    doclists = {str(it["id"]): [str(d) for d in it["relevant_docs"]] for it in sub}
    kq = {str(it["id"]): len(it["relevant_chunks"]) for it in c120in}
    la = loi_a_report(cands, doclists, kq, rrf3d2, WD, DELTA0)

    rep = {"thoi_gian_giay": round(time.time() - t0, 1),
           "ngoai_kho": {
               "so_id_chi_muc": len(idx_ids), "so_id_chi_muc_ngoai_kho": idx_ngoai,
               "so_id_pool": len(pool_ids), "so_id_pool_ngoai_kho": pool_ngoai,
               "so_cap_qdoc_bai": cap_bad, "TB_qdoc_ngoai_kho_cau": round(cap_bad / n, 2),
               "so_cau_anh_huong": cau_anh, "so_chunk_cua_doc_ngoai_kho": chunk_bad,
               "c120in": ref_ngoai(c120in), "cf4": ref_ngoai(cf4),
               "bad_vi28": len(bad)},
           "loi_a": la}
    write(OUT / "count_report.json", rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- stage sanity/build

def stage_sanity() -> None:
    files = chain_files(IN1_28, FILE_NAMES28)
    run_chain(SANITY_NAME, SANITY_DELTA, files, "text", OUT)
    cmp_rep = core.compare_subs(read(OUT / f"submission_{SANITY_NAME}.json"), str(SUB28))
    write(OUT / "sanity_report.json", cmp_rep)
    print(json.dumps(cmp_rep, ensure_ascii=False), flush=True)
    require_sanity(cmp_rep)


def stage_build() -> None:
    if (OUT / "build_report_vi29.json").exists() and (OUT / f"submission_{NAME}.zip").exists():
        print("build da xong (resume): bo qua", flush=True)
        return
    bad = v28.load_bad_ids()
    assert len(bad) == 16079, len(bad)
    corpus = load_corpus_ids()
    loc_rep = write_inputs(bad, IN2, corpus)
    files = chain_files(IN2, FILE_NAMES)
    delta, attempts = find_delta(files, "doc_text", MAX_ZIP, DELTA0, PROBE)
    r = chain_zip_check(NAME, delta, MAX_ZIP, files, "doc_text", validate=True)
    assert r["code"] == 0, r
    clean = assert_clean(read(OUT / f"submission_{NAME}.json"), bad, corpus)
    fp = fingerprint(OUT / f"submission_{NAME}.zip")
    write(OUT / "build_report_vi29.json",
          {"name": NAME, "delta": delta, "attempts": attempts, "max_zip": MAX_ZIP,
           "loc_dau_vao": loc_rep, "assert_clean": clean, "validate": "ok",
           "dedup_mode": "doc_text", "zip": fp,
           "n_q_TB": r.get("n_q_TB"), "tong_chunk": r.get("tong_chunk")})
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
    ref = read(SUB28)
    rep29 = read(OUT / f"report_chain_{NAME}.json")
    rep28 = read(OUT28 / "report_chain_vi28.json")
    r29 = sub_row(sub, rep29, fingerprint(OUT / f"submission_{NAME}.zip"), rep29.get("delta"))
    r28 = sub_row(ref, rep28, fingerprint(OUT28 / "submission_vi28.zip"), rep28.get("delta"))
    dl29 = {str(it["id"]): [str(d) for d in it["relevant_docs"]] for it in sub}
    dl28 = {str(it["id"]): [str(d) for d in it["relevant_docs"]] for it in ref}
    ch28 = {str(it["id"]): {(str(c["doc_id"]), c["chunk_text"]) for c in it["relevant_chunks"]}
            for it in ref}
    doi = trung30 = chunk_moi = cau_chunk_moi = 0
    for it in sub:
        q = str(it["id"])
        docs = dl29.get(q, [])
        other = dl28.get(q, [])
        doi += docs != other
        trung30 += len(set(docs[:30]) & set(other[:30]))
        moi = {(str(c["doc_id"]), c["chunk_text"]) for c in it["relevant_chunks"]} - ch28.get(q, set())
        chunk_moi += len(moi)
        cau_chunk_moi += bool(moi)
    n = max(len(sub), 1)
    so_sanh = {"so_cau_doclist_thay_doi": doi, "TB_top30_doc_trung_vi28": round(trung30 / n, 2),
               "so_cap_chunk_moi_vs_vi28": chunk_moi,
               "TB_chunk_moi_cau": round(chunk_moi / n, 2),
               "so_cau_co_chunk_moi": cau_chunk_moi}
    br = read(OUT / "build_report_vi29.json") if (OUT / "build_report_vi29.json").exists() else {}
    rep = {"vi29": r29, "vi28_tham_chieu": r28, "so_sanh": so_sanh,
           "attempts": br.get("attempts", []), "assert_clean": br.get("assert_clean"),
           "validate": "ok"}
    write(OUT / "report_vi29.json", rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- stage tokmeasure

def percentile(v_sorted: list, p: float):
    if not v_sorted:
        return None
    k = (len(v_sorted) - 1) * p / 100.0
    f = int(k)
    c = min(f + 1, len(v_sorted) - 1)
    return round(v_sorted[f] + (v_sorted[c] - v_sorted[f]) * (k - f), 1)


def dist(vals: list) -> dict:
    v = sorted(vals)
    if not v:
        return {"n": 0}
    return {"n": len(v), "min": v[0], "TB": round(sum(v) / len(v), 1), "p50": percentile(v, 50),
            "p90": percentile(v, 90), "max": v[-1]}


def load_queries(path=None) -> dict:
    """{str(id): query} tu query.parquet."""
    import pyarrow.parquet as pq
    d = pq.read_table(str(path or QUERY_PARQUET)).to_pydict()
    return {str(i): q for i, q in zip(d["id"], d["query"])}


def load_docs_wanted(paths, wanted: set) -> dict:
    """{id: text} cho id trong wanted, doc stream tung file JSONL.zst {id, idx, text};
    giu ban TRUOC khi trung id giua cac file."""
    from rerank_run import read_docs
    docs = {}
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        for r in read_docs(str(p)):
            if r["id"] in wanted and r["id"] not in docs:
                docs[r["id"]] = r["text"]
    return docs


def export_missing_docs(missing: set, out_zst: Path) -> dict:
    """Xuat text doc thieu tu chi muc vi25 bang export_docs (cache tuong minh tu ids.npy).
    Tra {yeu_cau, co_idx, ghi (doc doc duoc tu out_zst)}."""
    import numpy as np
    ids = [int(x) for x in np.load(IDX_IDS)]
    pos = {d: i for i, d in enumerate(ids)}
    rows = [[d, pos[d], 0.0] for d in sorted(missing) if d in pos]
    TOK_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = TOK_DIR / "missing_cache.json"
    write(cache_path, {"tok": rows})
    run("export_docs.py", "--index", ROOT / "data/index/vi25", "--cache", cache_path,
        "--n", max(len(rows), 1), "--chars", "200000", "--no-title",
        "--out-docs", out_zst, "--out-cands", TOK_DIR / "missing_cands.json")
    ghi = len(load_docs_wanted([out_zst], set(missing))) if out_zst.exists() else 0
    return {"yeu_cau": len(missing), "co_idx": len(rows), "ghi": ghi,
            "out": str(out_zst)}


class TokenMeter:
    """Dem token XLM-R (bge-reranker-v2-m3) cho cap (query, window), mo phong xu ly
    truncation="only_second", max_length=MAX_LEN cua rerank_run.score_pairs."""

    def __init__(self, kind: str, tok_full, tok_trunc, sp1: int, sp2: int, info: dict):
        self.kind = kind
        self.tok_full = tok_full
        self.tok_trunc = tok_trunc
        self.sp1 = sp1
        self.sp2 = sp2
        self.info = info

    def _len(self, tok, q, w=None) -> int:
        if self.kind == "transformers":
            return len(tok(q, w)["input_ids"] if w is not None else tok(q)["input_ids"])
        return len(tok.encode(q, w).ids if w is not None else tok.encode(q).ids)

    def _trunc_len(self, q: str, w: str):
        if self.tok_trunc is None:
            return None
        if self.kind == "transformers":
            return len(self.tok_trunc(q, w, truncation="only_second", max_length=MAX_LEN)["input_ids"])
        return len(self.tok_trunc.encode(q, w).ids)

    def measure(self, q: str, w: str) -> tuple:
        """Tra (nq_content, nw_content, pair_full, cut_window_tokens)."""
        nq = max(self._len(self.tok_full, q) - self.sp1, 0)
        nw = max(self._len(self.tok_full, w) - self.sp1, 0)
        full = self._len(self.tok_full, q, w)
        trunc = self._trunc_len(q, w)
        cut = max(full - trunc, 0) if trunc is not None else min(nw, max(full - MAX_LEN, 0))
        return nq, nw, full, cut


def load_tokenizer() -> tuple:
    """Tokenizer bge-reranker-v2-m3: transformers neu co san file tokenizer; khong thi goi
    `tokenizers` (cai ban ghim neu thieu) + tokenizer.json (chi tai tokenizer, khong weights).
    Tra (TokenMeter, info dict)."""
    info = {}
    try:
        from transformers import AutoTokenizer
        for cand in (TOK_DIR / "model", ROOT / "hf" / TOK_MODEL.replace("/", "_")):
            if cand.exists():
                tok = AutoTokenizer.from_pretrained(str(cand))
                sp1 = len(tok("")["input_ids"])
                sp2 = len(tok("", "")["input_ids"])
                info = {"nguon": f"transformers+{cand}", "model": TOK_MODEL}
                return TokenMeter("transformers", tok, tok, sp1, sp2, info), info
    except ImportError:
        pass
    try:
        import tokenizers
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", TOKENIZERS_PIN], check=True)
        import tokenizers
    if not TOK_JSON.exists():
        TOK_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(TOK_URL, str(TOK_JSON))
    from tokenizers import Tokenizer
    tok_full = Tokenizer.from_file(str(TOK_JSON))
    tok_trunc = Tokenizer.from_file(str(TOK_JSON))
    try:
        tok_trunc.enable_truncation(max_length=MAX_LEN, strategy="only_second", direction="right")
    except Exception:
        tok_trunc = None  # mo phong only_second bang cong thuc trong TokenMeter
    sp1 = len(tok_full.encode("").ids)
    sp2 = len(tok_full.encode("", "").ids)
    info = {"nguon": "tokenizers+tokenizer.json", "tokenizers": tokenizers.__version__,
            "file": str(TOK_JSON), "url": TOK_URL, "model": TOK_MODEL,
            "truncation": f"only_second/{MAX_LEN} (tokenizer)" if tok_trunc else f"only_second/{MAX_LEN} (cong thuc)"}
    return TokenMeter("tokenizers", tok_full, tok_trunc, sp1, sp2, info), info


def stage_tokmeasure() -> None:
    t0 = time.time()
    from docwin_run import build_docwin_pairs
    qtext = load_queries()
    cands = {str(q): v for q, v in core.load_json(IN1_28 / FILE_NAMES28["cands"]).items()}
    rng = random.Random(TOK_SEED)
    qids = sorted((q for q in cands if q in qtext), key=lambda x: int(x))
    chon = rng.sample(qids, TOK_N_Q)
    cap_doc = {}
    for q in chon:
        docs, seen = [], set()
        for r in sorted(cands[q], key=lambda r: (r[1], r[2])):
            d = int(r[0])
            if d in seen:
                continue
            seen.add(d)
            docs.append(d)
            if len(docs) >= TOK_N_DOC:
                break
        cap_doc[q] = docs
    wanted = {d for docs in cap_doc.values() for d in docs}
    docs_text = load_docs_wanted(DOCS_TXT, wanted)
    thieu_ids = wanted - set(docs_text)
    xuat_them = {"yeu_cau": 0, "co_idx": 0, "ghi": 0}
    missing_zst = TOK_DIR / "missing_docs.jsonl.zst"
    if thieu_ids:
        xuat_them = export_missing_docs(thieu_ids, missing_zst)
        docs_text.update(load_docs_wanted([missing_zst], thieu_ids))
    meter, tok_info = load_tokenizer()
    span_cache: dict = {}
    rec = []
    thieu_text = 0
    for q in chon:
        cand = [(d, i) for i, d in enumerate(cap_doc[q])]
        docs_map = {d: docs_text[d] for d in cap_doc[q] if d in docs_text}
        thieu_text += len(cap_doc[q]) - len(docs_map)
        pairs, _groups = build_docwin_pairs(qtext[q], cand, docs_map, TOK_N_DOC,
                                            WINDOW_WORDS, WINDOW_STRIDE, span_cache)
        for _qt, wtxt in pairs:
            rec.append(meter.measure(qtext[q], wtxt))
    nqs = [r[0] for r in rec]
    nws = [r[1] for r in rec]
    fulls = [r[2] for r in rec]
    cuts = [r[3] for r in rec]
    vuot = [f > MAX_LEN for f in fulls]
    pct_cat = [(c / w if w else 0.0) for c, w in zip(cuts, nws)]
    cat_vuot = [pc for pc, v in zip(pct_cat, vuot) if v]
    nguong = percentile(sorted(set(nqs)), 50) or 0

    def pct100(vals) -> float:
        return round(100.0 * (percentile(sorted(vals), 90) or 0.0), 1)

    def nhom(pred):
        idx = [i for i in range(len(rec)) if pred(nqs[i])]
        v = [vuot[i] for i in idx]
        return {"so_cap": len(idx),
                "query_tokens": dist([nqs[i] for i in idx]),
                "window_tokens": dist([nws[i] for i in idx]),
                "pair_tokens": dist([fulls[i] for i in idx]),
                "cap_vuot_512_pct": round(100.0 * sum(v) / max(len(idx), 1), 1),
                "window_bi_cat_pct": {"TB": round(100.0 * sum(pct_cat[i] for i in idx) / max(len(idx), 1), 1),
                                      "p90": pct100(pct_cat[i] for i in idx)}}

    rep = {"mau": {"seed": TOK_SEED, "so_cau": len(chon), "so_doc_mau": len(wanted),
                   "so_doc_co_text": len(docs_text), "so_cap_thieu_text": thieu_text,
                   "xuat_them_tu_chi_muc": xuat_them,
                   "so_cap_query_window": len(rec),
                   "TB_window_moi_doc": round(sum(len(v) for v in span_cache.values()) / max(len(span_cache), 1), 1),
                   "cua_so": {"window_words": WINDOW_WORDS, "window_stride": WINDOW_STRIDE},
                   "nguon_text": [str(p) for p in DOCS_TXT] + ([str(missing_zst)] if xuat_them["ghi"] else [])},
           "tokenizer": tok_info,
           "query_tokens": dist(nqs), "window_tokens": dist(nws), "pair_tokens": dist(fulls),
           "cap_vuot_512_pct": round(100.0 * sum(vuot) / max(len(rec), 1), 1),
           "window_bi_cat_pct": {"TB": round(100.0 * sum(pct_cat) / max(len(pct_cat), 1), 1),
                                 "p90": pct100(pct_cat),
                                 "chi_cap_vuot_512": {"TB": round(100.0 * sum(cat_vuot) / max(len(cat_vuot), 1), 1),
                                                      "p90": pct100(cat_vuot)}},
           "nhom_query": {"nguong_token": nguong,
                          "ngan": nhom(lambda x: x <= nguong),
                          "dai": nhom(lambda x: x > nguong)},
           "giay": round(time.time() - t0, 1)}
    write(OUT / "tokmeasure_report.json", rep)
    print(json.dumps(rep, ensure_ascii=False), flush=True)


# ---------------------------------------------------------------- status/cleanup/main

def status() -> None:
    print("time", time.strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    for p in sorted(OUT.glob("*.log")) + sorted(OUT.glob("*.json")):
        print(p.name, p.stat().st_size, time.ctime(p.stat().st_mtime), flush=True)
    print("locks", {p.name: p.read_text(errors="replace")
                    for p in (ROOT / "locks").glob("*.lock")}, flush=True)


def cleanup() -> None:
    lock = ROOT / "locks" / "heavy_job.lock"
    if lock.exists() and "vi29" in lock.read_text(errors="replace"):
        raise RuntimeError("vi29 lock still exists; inspect job before cleanup")
    name = "r2ai_vi29_run"
    q = subprocess.run(["schtasks", "/Query", "/TN", name, "/FO", "CSV", "/NH"],
                       capture_output=True, text=True)
    if not q.returncode:
        assert '"Running"' not in q.stdout, name + " still running"
        subprocess.run(["schtasks", "/Delete", "/TN", name, "/F"], check=True)
    print("CLEANUP DONE", "locks", {p.name: p.read_text(errors="replace")
                                    for p in (ROOT / "locks").glob("*.lock")}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["count", "sanity", "build", "report", "tokmeasure",
                                          "status", "cleanup"])
    args = parser.parse_args()
    if args.stage == "status":
        status()
        return
    if args.stage == "cleanup":
        cleanup()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with owned_lock(ROOT / "locks" / "heavy_job.lock", f"vi29 {args.stage} pid={os.getpid()}"):
        globals()["stage_" + args.stage]()
    print("DONE", args.stage, flush=True)


if __name__ == "__main__":
    main()
