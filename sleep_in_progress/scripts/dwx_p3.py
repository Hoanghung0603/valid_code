"""dwx_p3: docwin mở rộng cho doc của doclist_p3 chưa có ứng viên chunk (Phần B1).

Chunk ứng viên cũ (chunk_dw_u900_ext) chỉ có cho top-100 bge của cache_pool_rr và relevant_docs
của submission_split. Doc NEW lọt doclist_p3_t20 (và doc pool ngoài coverage) chưa có ứng viên
chunk -> cần chấm docwin (cửa sổ 300/250, U900, bge-reranker-v2-m3) đúng pipeline dwx.

pairs      (q, doc) trong các doclist (--doclist truyền được nhiều lần, gộp union) chưa có ứng viên
           chunk trong chunk_dw_u900_ext (coverage = top-100 cache_pool_rr ∪ relevant_docs
           submission_split) -> cands_docwin_p3.json {qid: [[doc_id, idx], ...]} cho docwin_run.py,
           cache_docwin_p3.json {qid: [[doc_id, idx, 0.0], ...]} cho export_docs.py --cache,
           dwx_p3_report.json (số cặp gộp + theo từng doclist, TB/câu, NEW vs pool ngoài coverage,
           doc thiếu idx).
winsample  lấy mẫu N cửa sổ CŨ từ docwin_ext.json.zst (logit tham chiếu) + text doc + query ->
           winsample.json: [{qid, doc_id, start, end, ref_logit, query, window}] — chấm lại trên
           z87 để kiểm logit khớp ~1e-2 (cùng model/cửa sổ/max-len như docwin cũ).
score      chấm (query, cửa sổ) của winsample bằng bge-reranker-v2-m3 (GPU) -> winscore.json
           (cùng dòng + logit mới).
wincheck   so logit chấm lại với ref: max/TB |sai|; ok nếu max <= tol (mặc định 1e-2).

Dùng:
  python scripts/dwx_p3.py pairs --doclist diag18\\doclist_p3_t20.json --doclist diag18\\doclist_p3_t16.json \
      --doclist diag18\\doclist_p3_nq.json --pool cache_pool_rr.json \
      --split diag12\\submission_split.json --idx-from diag18\\cache_pool3_rr.json diag18\\cache_fhyde3.json \
      --new diag18\\new_cands.json --out-dir diag18\\docwin_p3
  python scripts/dwx_p3.py winsample --docwin work\\docwin_ext\\docwin_ext.json.zst \
      --docs work\\docwin_ext\\docwin_ext.part000.jsonl.zst --queries work\\queries.jsonl --out winsample.json
  python scripts/dwx_p3.py score --in winsample.json --out winscore.json
  python scripts/dwx_p3.py wincheck --ref winsample.json --got winscore.json --report wincheck.json
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

TOP_N = 100  # top-N cache_pool_rr đã có docwin/chunk cũ
BANDS = ((1, 30, "1-30"), (31, 60, "31-60"), (61, 100, "61-100"), (101, 10 ** 9, "101+"))


def load_json(path: str):
    p = Path(path)
    if p.name.endswith(".zst"):
        import zstandard
        with open(p, "rb") as f:
            return json.loads(zstandard.ZstdDecompressor().stream_reader(f).read())
    return json.loads(p.read_text(encoding="utf-8"))


def band_of(rank: int) -> str:
    for a, b, k in BANDS:
        if a <= rank <= b:
            return k
    return BANDS[-1][2]


def norm_id(d):
    return int(d) if str(d).lstrip("-").isdigit() else d


def idx_lookup(caches: list) -> dict:
    """{str(doc_id): idx} — hit đầu tiên có idx khác None trong các cache nguồn."""
    out = {}
    for cache in caches:
        for hits in cache.values():
            for h in hits or []:
                if len(h) > 1 and h[1] is not None:
                    out.setdefault(str(h[0]), int(h[1]))
    return out


def build_pairs(doclist: dict, pool: dict, split_docs: dict, idx_of: dict,
                new_ids: dict, top_n: int = TOP_N) -> tuple[dict, dict, dict]:
    """(cands, cache, report) — (q, doc) trong doclist ngoài coverage; thứ tự = thứ tự doclist.

    coverage(q) = top-n cache_pool[q] ∪ relevant_docs split[q] (đã có ứng viên chunk_dw_u900_ext).
    """
    cands, cache = {}, {}
    pairs = pairs_new = pairs_old = n_no_idx = 0
    uniq: set = set()
    band_pairs = {k: 0 for _, _, k in BANDS}
    lens = []
    for q in sorted(doclist, key=lambda x: int(x) if str(x).isdigit() else x):
        top = {str(h[0]) for h in (pool.get(q) or [])[:top_n]}
        cov = top | {str(d) for d in split_docs.get(q, [])}
        nset = {str(x) for x in new_ids.get(q, [])}
        rows, m = [], []
        for r, d in enumerate(doclist[q], 1):
            s = str(d)
            if s in cov:
                continue
            i = idx_of.get(s)
            if i is None:
                n_no_idx += 1
                continue
            rows.append([norm_id(d), i])
            m.append([norm_id(d), i, 0.0])
            pairs += 1
            uniq.add(s)
            band_pairs[band_of(r)] += 1
            if s in nset:
                pairs_new += 1
            else:
                pairs_old += 1
        lens.append(len(rows))
        if rows:
            cands[q] = rows
            cache[q] = m
    n = max(len(doclist), 1)
    report = {"cau": len(doclist), "cap": pairs, "TB_cap_cau": pairs / n,
              "min_cap": min(lens) if lens else 0, "max_cap": max(lens) if lens else 0,
              "doc_day_du_trong_cands": sum(1 for x in lens if x),
              "doc_duy_nhat": len(uniq), "cap_NEW": pairs_new, "TB_NEW_cau": pairs_new / n,
              "cap_pool_ngoai_coverage": pairs_old, "TB_pool_ngoai_coverage_cau": pairs_old / n,
              "cap_theo_dai_hang_doclist": band_pairs,
              "cap_thieu_idx_khong_export_duoc": n_no_idx,
              "top_n": top_n}
    return cands, cache, report


def cmd_pairs(a) -> int:
    t0 = time.time()
    pool = {str(k): v for k, v in load_json(a.pool).items()}
    split = load_json(a.split)
    split_docs = {str(it["id"]): it["relevant_docs"] for it in split}
    new_ids = ({str(k): {str(d) for d in v} for k, v in load_json(a.new).items()} if a.new else {})
    caches = [pool] + [{str(k): v for k, v in load_json(p).items()} for p in a.idx_from]
    idx_of = idx_lookup(caches)
    cands, cache, rep = {}, {}, {"cac_doclist": {}}
    qs: set = set()
    for path in a.doclist:
        doclist = {str(k): v for k, v in load_json(path).items()}
        qs |= set(doclist)
        c, m, r = build_pairs(doclist, pool, split_docs, idx_of, new_ids, a.top_n)
        rep["cac_doclist"][path] = r
        merge_pairs(cands, cache, c, m)
    rep_gop = summarize_pairs(cands, cache, new_ids, max(len(qs), 1))
    rep = {**rep_gop, **rep, "doclist": a.doclist}
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cands_docwin_p3.json").write_text(json.dumps(cands), encoding="utf-8")
    (out_dir / "cache_docwin_p3.json").write_text(json.dumps(cache), encoding="utf-8")
    rep["thoi_gian_giay"] = round(time.time() - t0, 1)
    (out_dir / "dwx_p3_report.json").write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    print(f"ghi cands_docwin_p3.json + cache_docwin_p3.json vào {out_dir}", flush=True)
    return 0


def merge_pairs(cands: dict, cache: dict, c: dict, m: dict) -> None:
    """Gộp cands/cache của một doclist vào accumulator, bỏ trùng doc_id theo thứ tự xuất hiện."""
    for q, rows in c.items():
        seen = {r[0] for r in cands.get(q, [])}
        for r, mr in zip(rows, m[q]):
            if r[0] in seen:
                continue
            seen.add(r[0])
            cands.setdefault(q, []).append(r)
            cache.setdefault(q, []).append(mr)


def summarize_pairs(cands: dict, cache: dict, new_ids: dict, n_cau: int | None = None) -> dict:
    """Thống kê bản gộp: cap, TB/câu, NEW vs pool ngoài coverage, doc duy nhất."""
    n_q = n_cau or max(len(cands), 1)
    lens = [len(v) for v in cands.values()]
    n_new = 0
    uniq: set = set()
    for q, rows in cands.items():
        nset = new_ids.get(q, set())
        for r in rows:
            uniq.add(str(r[0]))
            if str(r[0]) in nset:
                n_new += 1
    cap = sum(lens)
    return {"gop": {"cau_co_cap": len(cands), "cap": cap, "TB_cap_cau": cap / n_q,
                    "min_cap": min(lens) if lens else 0, "max_cap": max(lens) if lens else 0,
                    "doc_duy_nhat": len(uniq), "cap_NEW": n_new, "TB_NEW_cau": n_new / n_q,
                    "cap_pool_ngoai_coverage": cap - n_new}}


def sample_windows(docwin: dict, docs: dict, qtext: dict, n: int, seed: int) -> list:
    """Lấy mẫu n cửa sổ có text: [{qid, doc_id, start, end, ref_logit, query, window}]."""
    pool = []
    for q in sorted(docwin, key=lambda x: int(x) if str(x).isdigit() else x):
        qt = qtext.get(str(q))
        if qt is None:
            continue
        for d in sorted(docwin[q], key=lambda x: str(x)):
            t = docs.get(str(d))
            if t is None:
                continue
            for s, e, lg in docwin[q][d]:
                pool.append((str(q), d, s, e, lg, qt, t[s:e]))
    rnd = random.Random(seed)
    if len(pool) > n:
        pool = rnd.sample(pool, n)
    return [{"qid": q, "doc_id": norm_id(d), "start": s, "end": e, "ref_logit": lg,
             "query": qt, "window": w} for q, d, s, e, lg, qt, w in pool]


def cmd_winsample(a) -> int:
    t0 = time.time()
    docwin = load_json(a.docwin)
    docs = {str(r["id"]): r["text"] for r in _iter_docs(a.docs)}
    qtext = {str(r["id"]): r["query"] for r in
             (json.loads(l) for l in Path(a.queries).read_text(encoding="utf-8").splitlines() if l.strip())}
    rows = sample_windows(docwin, docs, qtext, a.n, a.seed)
    Path(a.out).write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    print(f"lấy mẫu {len(rows)} cửa sổ từ {a.docwin} -> {a.out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def _iter_docs(paths) -> iter:
    """Yield {id, text,...} từ một hoặc nhiều JSONL(.zst)."""
    import zstandard
    for p in paths:
        pp = Path(p)
        if pp.name.endswith(".zst"):
            with open(pp, "rb") as f:
                raw = zstandard.ZstdDecompressor().stream_reader(f).read()
            lines = raw.splitlines()
        else:
            lines = pp.read_bytes().splitlines()
        for l in lines:
            if l.strip():
                yield json.loads(l)


def cmd_score(a) -> int:
    """Chấm (query, cửa sổ) bằng bge-reranker-v2-m3 — chỉ chạy trên máy GPU (z87)."""
    t0 = time.time()
    rows = json.loads(Path(a.in_).read_text(encoding="utf-8"))
    from rerank_run import load_model, score_pairs
    tok, model = load_model(a.model)
    pairs = [(r["query"], r["window"]) for r in rows]
    scores = score_pairs(model, tok, pairs, a.max_len, a.batch)
    out = []
    for r, sc in zip(rows, scores):
        out.append({k: r[k] for k in ("qid", "doc_id", "start", "end", "ref_logit")})
        out[-1]["logit"] = round(float(sc), 4)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"chấm {len(out)} cửa sổ -> {a.out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_wincheck(a) -> int:
    ref = json.loads(Path(a.ref).read_text(encoding="utf-8"))
    got = json.loads(Path(a.got).read_text(encoding="utf-8"))
    gm = {(str(r["qid"]), r["doc_id"], r["start"], r["end"]): r["logit"] for r in got}
    diffs, n_ok = [], 0
    for r in ref:
        key = (str(r["qid"]), r["doc_id"], r["start"], r["end"])
        if key not in gm:
            continue
        d = abs(float(gm[key]) - float(r["ref_logit"]))
        diffs.append(d)
        n_ok += 1
    rep = {"tol": a.tol, "so_cap": n_ok, "so_mau": len(ref), "thieu": len(ref) - n_ok,
           "max_sai": max(diffs) if diffs else None,
           "TB_sai": sum(diffs) / len(diffs) if diffs else None,
           "ok": bool(diffs) and max(diffs) <= a.tol and n_ok == len(ref)}
    Path(a.report).write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    print(f"ghi {a.report}", flush=True)
    return 0 if rep["ok"] else 2


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("pairs")
    p.add_argument("--doclist", action="append", required=True, metavar="PATH",
                   help="doclist JSON {qid: [doc_id,...]} — truyền nhiều lần để gộp cặp (union)")
    p.add_argument("--pool", required=True, help="cache_pool_rr.json")
    p.add_argument("--split", required=True, help="submission_split.json (coverage)")
    p.add_argument("--idx-from", nargs="*", default=[], help="cache nguồn lấy idx")
    p.add_argument("--new", default="", help="new_cands.json (đếm cặp NEW)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-n", type=int, default=TOP_N)
    p.set_defaults(fn=cmd_pairs)
    p = sp.add_parser("winsample")
    p.add_argument("--docwin", required=True, help="docwin_ext.json.zst (logit cửa sổ cũ)")
    p.add_argument("--docs", required=True, nargs="+", help="JSONL.zst text doc (part)")
    p.add_argument("--queries", required=True, help="queries.jsonl {id, query}")
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=20261014)
    p.set_defaults(fn=cmd_winsample)
    p = sp.add_parser("score")
    p.add_argument("--in", dest="in_", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    p.add_argument("--max-len", type=int, default=512)
    p.add_argument("--batch", type=int, default=16)
    p.set_defaults(fn=cmd_score)
    p = sp.add_parser("wincheck")
    p.add_argument("--ref", required=True, help="winsample.json (ref_logit)")
    p.add_argument("--got", required=True, help="winscore.json (logit)")
    p.add_argument("--report", required=True)
    p.add_argument("--tol", type=float, default=1e-2)
    p.set_defaults(fn=cmd_wincheck)
    a = ap.parse_args()
    sys.exit(a.fn(a))


if __name__ == "__main__":
    main()
