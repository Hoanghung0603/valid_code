r"""Loi CPU cua runner dau-cuoi private (pha 1) - ham thuan, co unit test.

Tai hien cong thuc vi34 tren tap cau TU Y (id bat ky, so cau bat ky), tu query.parquet +
chi muc bat ky -> submission/zip:
  pool  = BM25 top-200 U dense bge-m3 top-200 (chi muc); rerank bge-reranker-v2-m3 TOAN pool
  8B    = Qwen3-Reranker-8B cho toan doclist + 30 hang bien (quy tac q8b33, lap den khi phu du)
  fuse  = rrf3d2 = RRF k60 (bge 1, 8B 1, dense 2) -> zd chunk; fhyde = bge 1, 8B 2, dense 2, HyDE 1
  n_q   = so doc logit bge > -2.0 kep [40, 400] (union_sub.adaptive_n)
  doclist = n_q doc dau fhyde, KHONG ref_chunks cf4 (PIPELINE_v4 §10.2.7)
  chunk = docwin 300/250 moi doc doclist (+ top-100 bge cho k_q), U900, s = zc + 4*zd, b_cov 1,
          dedup (doc_id, text); k_q = quy tac noi tai tai hien c120in (intrinsic_kq);
          delta lon nhat cho zip LZMA <= 99.000.000 B
  loc   = bad 16.079 id + id ngoai links_corpus_vi_en_zh.parquet; assert_clean KHONG neo id 1..1200

Chain dung nguyen vi29.chain_build_v29 (dedup "doc_text", ref_chunks {}) voi vi23.W_Q8B_FHYDE = 2.0
de cong thuc khong can 2 ban; k_q truyen vao la k_q noi tai. Moi file moi nam rieng; khong sua
script vi*/q8b*/diag* hien co.
"""
import hashlib
import json
import time
from pathlib import Path

THR_N, LO_N, HI_N = -2.0, 40, 400
WD, B_COV = 4.0, 1.0
K_RRF = 60
W_BGE, W_Q8B, W_DENSE, W_HYDE = 1.0, 1.0, 2.0, 1.0
W_Q8B_FHYDE = 2.0
MAX_ZIP = 99_000_000
KQ_CAP = 120
Q8B_MARGIN = 30
TOP_BGE = 100
SMOKE_N = 20


# ---------------------------------------------------------------- tien ich file/manifest

def fingerprint(p) -> dict:
    """{path, bytes, sha256} cua file (sha theo tung khoi 1MB)."""
    p = Path(p)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"path": str(p), "bytes": p.stat().st_size, "sha256": h.hexdigest()}


CURRENT_RUN_ARGS: dict = {}   # runner_private.dat; manifest ghi lai de resume dung khi doi tham so


def write_manifest(out_dir, stage: str, params: dict, inputs: list, outputs: list,
                   seconds: float, extra: dict | None = None, machine: str = "") -> Path:
    """Ghi manifest_<stage>.json: tham so, sha256 moi dau vao/ra, thoi gian, may chay."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def fp(p):
        p = Path(p)
        return fingerprint(p) if p.exists() else {"path": str(p), "missing": True}

    man = {"stage": stage, "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "machine": machine,
           "params": params, "run_args": dict(CURRENT_RUN_ARGS),
           "inputs": [fp(p) for p in inputs], "outputs": [fp(p) for p in outputs],
           "seconds": round(seconds, 1)}
    if extra:
        man.update(extra)
    dest = out_dir / f"manifest_{stage}.json"
    dest.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    return dest


def load_json(path):
    p = Path(path)
    if p.name.endswith(".zst"):
        import zstandard
        with p.open("rb") as f:
            return json.loads(zstandard.ZstdDecompressor().stream_reader(f).read())
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path, obj) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(obj, ensure_ascii=False)
    p.write_text(raw, encoding="utf-8")
    return len(raw.encode("utf-8"))


# ---------------------------------------------------------------- truy van

def load_queries(path) -> list:
    """[(id int, query str)] tu parquet (id, query) hoac JSONL {id, query}; trung id lay ban dau."""
    p = Path(path)
    rows = []
    if p.suffix == ".parquet":
        import pandas as pd
        df = pd.read_parquet(p)
        rows = [(int(i), str(s)) for i, s in zip(df["id"].astype(int), df["query"])]
    else:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            rows.append((int(r["id"]), str(r.get("query", r.get("zh", "")))))
    out, seen = [], set()
    for i, q in rows:
        if i in seen or not q.strip():
            continue
        seen.add(i)
        out.append((i, q))
    return out


def evenly_spaced(items: list, n: int) -> list:
    """n phan tu rai deu tren danh sach da sap (vi tri round(i*(N-1)/(n-1))), n >= 2."""
    if n >= len(items):
        return list(items)
    if n < 2:
        raise ValueError("n phai >= 2")
    s = sorted(items)
    m = len(s) - 1
    return [s[round(i * m / (n - 1))] for i in range(n)]


# ---------------------------------------------------------------- pool / fuse / doclist

def pool_union(bm25_hits: list, dense_hits: list, top: int = 200) -> list:
    """[[doc_id, idx], ...] = BM25 top-top truoc, them doc dense top-top chua co (giu thu tu)."""
    out, seen = [], set()
    for hits in (bm25_hits or [])[:top]:
        d = str(hits[0])
        if d in seen:
            continue
        seen.add(d)
        out.append([hits[0], hits[1]])
    for hits in (dense_hits or [])[:top]:
        d = str(hits[0])
        if d in seen:
            continue
        seen.add(d)
        out.append([hits[0], hits[1]])
    return out


def pool_union_multi(sources: list) -> list:
    """Hop nhieu nguon [(hits, top), ...] theo thu tu nguon, bo trung giu lan dau (= pool_union mo rong).

    pool_union(bm, dn, top) == pool_union_multi([(bm, top), (dn, top)])."""
    out, seen = [], set()
    for hits, top in sources:
        for h in (hits or [])[:top]:
            d = str(h[0])
            if d in seen:
                continue
            seen.add(d)
            out.append([h[0], h[1]])
    return out


def union_pairs(*maps: dict) -> dict:
    """Hop {qid: [doc, ...]} nhieu map theo thu tu, bo trung (giu lan dau). Dung cho cap 8B / docwin
    cua hai pool mode: cung mot cap (cau, doc) chi cham mot lan."""
    out: dict = {}
    for m in maps:
        for q, docs in m.items():
            lst = out.setdefault(q, [])
            seen = {str(d) for d in lst}
            for d in docs:
                if str(d) not in seen:
                    lst.append(d)
                    seen.add(str(d))
    return out


POOL_MODES = ("base", "p1")


def pool_for_mode(mode: str, bm25_k300: list, dense200: list, hyde200: list) -> list:
    """base = BM25-200 U dense-200 (dinh nghia hien tai); p1 = BM25-300 U dense-200 U HyDE-200.
    base la tap con cua p1 (cung thu tu uu tien dau)."""
    if mode == "base":
        return pool_union_multi([(bm25_k300, 200), (dense200, 200)])
    if mode == "p1":
        return pool_union_multi([(bm25_k300, 300), (dense200, 200), (hyde200, 200)])
    raise ValueError(f"pool mode la {mode}")


def sort_by_score(hits: list) -> list:
    """[[doc_id, idx, score], ...] sap score giam dan; score None xuong cuoi, on dinh."""
    rows = [list(h) for h in hits or []]
    return sorted(rows, key=lambda h: (h[2] is None, 0.0 if h[2] is None else -h[2]))


def fuse_rrf(pool: dict, q8b: dict, dense: dict, hyde: dict, w8f: float = W_Q8B_FHYDE):
    """(rrf3d2, fhyde) dung fuse3: rrf3d2 = (bge 1, 8B 1, dense 2); fhyde = (bge 1, 8B w8f, dense 2, HyDE 1).

    Giong vi23.chain_build: nguon thu 4 (HyDE) truyen qua tham so thu 6 cua fuse3 (ten bm25).
    Ung vien = doc cua pool; doc khong trong pool khong vao ket qua."""
    from fuse3 import fuse3 as fuse3_fn
    rrf3d2 = fuse3_fn(pool, q8b, dense, (W_BGE, W_Q8B, W_DENSE, 0.0), K_RRF, None, None)
    fhyde = fuse3_fn(pool, q8b, dense, (W_BGE, w8f, W_DENSE, W_HYDE), K_RRF, hyde, None)
    return rrf3d2, fhyde


def n_q_for(pool_hits: list, thr: float = THR_N, lo: int = LO_N, hi: int = HI_N) -> int:
    """n_q = so doc logit bge > thr, kep [lo, hi] (union_sub.adaptive_n)."""
    from union_sub import adaptive_n
    return adaptive_n(pool_hits or [], thr, lo, hi)


def doclist_for(fhyde_hits: list, n: int) -> list:
    """n doc dau theo fhyde, khong trung (vi23.first_n_docs). Khong them doc nguon ref_chunks."""
    import vi23 as core
    return core.first_n_docs(fhyde_hits or [], n)


def top_bge(pool_hits: list, k: int = TOP_BGE) -> list:
    """k doc dau theo logit bge = thu tu cua pool (pool da sap logit giam)."""
    import vi23 as core
    return core.hit_ids(pool_hits or [], k)


def q8b_pairs_for(fhyde: dict, pool: dict, doclists: dict, scored: dict | None = None,
                  margin: int = Q8B_MARGIN) -> dict:
    """{qid: [doc_id...]} can cham 8B: top-(len(doclist)+margin) cua fhyde, chi doc trong pool,
    bo doc da co diem 8B (scored). Quy tac q8b33 tren bo khong co cache cu."""
    sc = {str(q): {str(h[0]) for h in v or [] if h[2] is not None} for q, v in (scored or {}).items()}
    po = {str(q): {str(h[0]) for h in v or []} for q, v in pool.items()}
    out = {}
    for q, hits in fhyde.items():
        q = str(q)
        top = len(doclists.get(q, [])) + margin
        ps, os_, seen, keep = po.get(q, set()), sc.get(q, set()), set(), []
        for h in (hits or [])[:top]:
            d = str(h[0])
            if d in seen or d not in ps or d in os_:
                continue
            seen.add(d)
            keep.append(int(h[0]) if str(h[0]).lstrip("-").isdigit() else h[0])
        out[q] = keep
    return out


def merge_q8b(old: dict, new: dict) -> tuple:
    """Gop cache 8B (q8b31.merge_q8b): hit moi score None bi bo, trung giu diem cu, sap giam."""
    from q8b31 import merge_q8b as _m
    return _m(old, new)


# ---------------------------------------------------------------- k_q noi tai

def intrinsic_kq(rows: list, fused_hits: list, allowed, cap: int = KQ_CAP, wd: float = WD,
                 dedup_mode: str = "doc_text", mode: int = 0) -> int:
    """k_q noi tai tai hien submission_c120in (diag14.build_sub mode c120in, cap=120):
    order_q tren ung vien chunk U900 cua lop docwin top-100 bge (wd=4, zc + 4*zd) -> lay cap
    dong dau -> loc doc trong doclist -> k_q = so dong con lai.

    mode 0: cat 120 TRUOC roi loc doclist (dung nhu c120in goc).
    mode 1: loc doclist truoc roi cat 120 (bien the don gian, dung de do).
    dedup_mode "doc_text" = luat vi29 (dung); "text" = luat cu cua c120in."""
    import vi29 as v29
    allowed = {str(d) for d in (allowed or [])}
    order = v29.order_q_docdedup(rows or [], fused_hits or [], None, dedup_mode, wd)
    if mode == 0:
        return sum(1 for c in order[:cap] if str(c["doc_id"]) in allowed)
    return min(cap, sum(1 for c in order if str(c["doc_id"]) in allowed))


# ---------------------------------------------------------------- chain

class _Vi34Weights:
    """Dat vi23.W_Q8B_FHYDE = w8f (fhyde) va NQ_FUSED = None trong luc chay chain; khoi phuc sau."""

    def __init__(self, w8f: float = W_Q8B_FHYDE):
        self.w8f = w8f

    def __enter__(self):
        import vi23 as core
        self._old = (getattr(core, "W_Q8B_FHYDE", None), getattr(core, "NQ_FUSED", None))
        core.W_Q8B_FHYDE = self.w8f
        core.NQ_FUSED = None
        return self

    def __exit__(self, *exc):
        import vi23 as core
        core.W_Q8B_FHYDE, core.NQ_FUSED = self._old
        return False


def chain_build_runner(pool: dict, q8b: dict, dense: dict, hyde: dict, cands: dict, kq: dict,
                       delta: int, thr: float = THR_N, lo: int = LO_N, hi: int = HI_N,
                       wd: float = WD, b_cov: float = B_COV, w8f: float = W_Q8B_FHYDE) -> tuple:
    """Chain vi34 voi ref_chunks = {} (bo cf4) va k_q truyen vao (noi tai). Dung nguyen
    vi29.chain_build_v29 dedup "doc_text". Tra (rrf3d2, fhyde, sub, rep)."""
    import vi29 as v29
    with _Vi34Weights(w8f):
        return v29.chain_build_v29(pool, q8b, dense, hyde, cands, kq, {}, thr, lo, hi, wd,
                                   b_cov, delta, "doc_text")


def run_chain_runner(out_dir, name: str, delta: int, pool: dict, q8b: dict, dense: dict,
                     hyde: dict, cands: dict, kq: dict, **kw) -> dict:
    """Chay chain_build_runner, ghi submission_<name>.json + doclist + report_chain_<name>.json
    (dung duoc voi vi23.check_zip). Tra report."""
    t0 = time.time()
    rrf3d2, fhyde, sub, rep = chain_build_runner(pool, q8b, dense, hyde, cands, kq, delta, **kw)
    out_dir = Path(out_dir)
    fp = out_dir / f"submission_{name}.json"
    write_json(fp, sub)
    write_json(out_dir / f"doclist_{name}.json", {str(it["id"]): it["relevant_docs"] for it in sub})
    rep = dict(rep)
    rep.update({"name": name, "json_bytes": fp.stat().st_size, "thoi_gian_giay": round(time.time() - t0, 1)})
    write_json(out_dir / f"report_chain_{name}.json", rep)
    return rep


def chain_zip_check_runner(out_dir, name: str, delta: int, max_zip: int, files: dict, **kw) -> dict:
    """Mot lan thu chain + validate (neu --validate) + zip LZMA + check_zip. Tra code/zip_bytes."""
    import subprocess
    import sys
    out_dir = Path(out_dir)
    S = Path(__file__).resolve().parent
    pool = load_json(files["pool"])
    q8b = load_json(files["q8b"]) if Path(files["q8b"]).exists() else {}
    dense = load_json(files["dense"])
    hyde = load_json(files["hyde"]) if files.get("hyde") and Path(files["hyde"]).exists() else {}
    cands = {str(q): v for q, v in load_json(files["cands"]).items()}
    kq = {str(q): v for q, v in load_json(files["kq"]).items()}
    run_chain_runner(out_dir, name, delta, pool, q8b, dense, hyde, cands, kq, **kw)
    if kw.get("validate"):
        subprocess.run([sys.executable, str(S / "evaluate.py"), "validate",
                        str(out_dir / f"submission_{name}.json"),
                        "--n-queries", str(len(pool))], check=True)
    subprocess.run([sys.executable, str(S / "zip_submission.py"), "--json",
                    str(out_dir / f"submission_{name}.json"), "--zip",
                    str(out_dir / f"submission_{name}.zip"), "--method", "lzma"], check=True)
    import vi23 as core
    code = core.check_zip(str(out_dir / f"submission_{name}.json"),
                          str(out_dir / f"submission_{name}.zip"),
                          str(out_dir / f"report_chain_{name}.json"), name, max_zip=max_zip)
    rep = load_json(out_dir / f"report_chain_{name}.json")
    entry = rep.get("zip", {}).get(name, {})
    return {"code": code, "next_delta": entry.get("next_delta"), "zip_bytes": entry.get("zip_bytes"),
            "json_bytes": rep.get("json_bytes"), "tong_chunk": rep.get("tong_chunk"),
            "hist_kcap": rep.get("hist_kcap"), "n_q_TB": rep.get("n_q_TB")}


def find_delta_runner(attempt, delta0: int, max_zip: int = MAX_ZIP, max_tries: int = 16) -> tuple:
    """Delta LON NHAT voi zip <= max_zip (thuat toan vi29.find_delta, attempt la closure).

    attempt(delta) -> {"code": 0 ok | 3 vuot (co next_delta), "next_delta", "zip_bytes",
    "json_bytes", "tong_chunk", "hist_kcap"}; moi lan thu duoc cache boi attempt.
    Tra (delta, [{delta, code, zip_bytes}...])."""
    from cx130 import largest_delta_within
    tried = {}

    def attempt_cached(d):
        if d not in tried:
            tried[d] = attempt(d)
        return tried[d]

    def fits(d):
        return attempt_cached(d)["code"] == 0

    def grew(d, base):
        return tried[d].get("tong_chunk") != tried[base].get("tong_chunk")

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
            break
        hi = up
        while hi - lo > 1 and len(tried) < max_tries:
            mid = (lo + hi) // 2
            if fits(mid) and grew(mid, lo):
                lo = mid
            else:
                hi = mid
        break
    while lo + 1 not in tried and len(tried) < max_tries:
        if fits(lo + 1) and grew(lo + 1, lo):
            lo += 1
        else:
            break
    return lo, [{"delta": k, "code": v["code"], "zip_bytes": v["zip_bytes"]}
                for k, v in sorted(tried.items())]


# ---------------------------------------------------------------- assert_clean / so sanh

def assert_clean_runner(sub: list, bad, corpus=None, expect_ids=None) -> dict:
    """Kiem tra bai: id rieng (khong neo 1..1200); 0 bad/ngoi kho; moi chunk trong doclist;
    khong trung (doc_id, chunk_text) trong mot cau. expect_ids neu co phai khop het."""
    bs = {str(i) for i in (bad or [])}
    cs = {str(i) for i in corpus} if corpus is not None else None
    ids = [it["id"] for it in sub]
    assert len(set(map(str, ids))) == len(ids), "id trung"
    if expect_ids is not None:
        assert set(map(str, ids)) == {str(i) for i in expect_ids}, "khong khop tap id mong doi"
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


def compare_subs_subset(got: list, ref: list, top: int = 30) -> dict:
    """So 2 bai tren cung tap id: top-k doc trung, doclist trung, chunk (doc_id,text) trung, n_q."""
    gr = {str(it["id"]): it for it in got}
    rr = {str(it["id"]): it for it in ref}
    qs = sorted(set(gr) & set(rr), key=lambda x: int(x))
    top_trung = jac = chung = chunk_moi = n_g = n_r = 0
    for q in qs:
        g, r = gr[q], rr[q]
        gd = [str(d) for d in g["relevant_docs"]]
        rd = [str(d) for d in r["relevant_docs"]]
        top_trung += len(set(gd[:top]) & set(rd[:top]))
        a, b = set(gd), set(rd)
        jac += len(a & b) / max(len(a | b), 1)
        gc = {(str(c["doc_id"]), c["chunk_text"]) for c in g["relevant_chunks"]}
        rc = {(str(c["doc_id"]), c["chunk_text"]) for c in r["relevant_chunks"]}
        chung += len(gc & rc)
        chunk_moi += len(gc - rc)
        n_g += len(g["relevant_chunks"])
        n_r += len(r["relevant_chunks"])
    n = max(len(qs), 1)
    return {"so_cau": len(qs), f"TB_top{top}_doc_trung": round(top_trung / n, 2),
            "TB_doclist_jaccard": round(jac / n, 4),
            "TB_doc_got": round(sum(len(gr[q]["relevant_docs"]) for q in qs) / n, 2),
            "TB_doc_ref": round(sum(len(rr[q]["relevant_docs"]) for q in qs) / n, 2),
            "TB_chunk_got": round(n_g / n, 2), "TB_chunk_ref": round(n_r / n, 2),
            "TB_chunk_trung_nguyen_van": round(chung / n, 2),
            "TB_chunk_moi_cau": round(chunk_moi / n, 2)}


# ---------------------------------------------------------------- bench

def sample_cache_pairs(cache: dict, n: int, seed: int = 42, top: int = 0) -> list:
    """Mau (qid, doc_id, score) tu cache {qid: [[doc, idx, score], ...]} co diem."""
    import random
    cand = []
    for q, hits in cache.items():
        for h in (hits or [])[:top or len(hits)]:
            if len(h) > 2 and h[2] is not None:
                cand.append((str(q), h[0], float(h[2])))
    cand = sorted(set(cand))
    random.Random(seed).shuffle(cand)
    return cand[:n]


def sample_docwin_rows(wins: dict, n: int, seed: int = 43) -> list:
    """Mau (qid, doc_id, start, end, logit) tu docwin {qid: {doc: [[s,e,logit], ...]}}."""
    import random
    cand = []
    for q, by_d in wins.items():
        for d, rows in (by_d or {}).items():
            for s, e, lg in rows:
                cand.append((str(q), d, s, e, float(lg)))
    cand = sorted(set(cand))
    random.Random(seed).shuffle(cand)
    return cand[:n]


def corr(xs: list, ys: list) -> dict:
    """Pearson/Spearman/|lech| (q8b31.corr)."""
    from q8b31 import corr as _c
    return _c(xs, ys)
