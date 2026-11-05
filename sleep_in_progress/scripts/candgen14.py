"""candgen14: sinh ứng viên doc NGOÀI pool hiện tại và dựng 2 bài chẩn đoán (chạy CPU trên b660).

Nguồn ứng viên (mỗi nguồn = cache {qid: [[doc_id, idx, score], ...]} giảm dần, rank 1-based):
  pool2  cache_pool2_rr.json (pool ∪ dense top-300 ∪ BM25 top-300, có logit rerank)
  hyde   dense_hyde_top200.json (dense bge-m3 toàn kho, truy vấn HyDE)
  reason dense_reason_top200.json (dense bge-m3 toàn kho, truy vấn reason)
  dense  dense_full_top400.json (dense bge-m3 toàn kho, lấy --dense-k đầu)
  bm25   cache_vi_k300.json (BM25 toan kho, lay --bm25-k dau)

1. stats  NEW(q) = (hợp nguồn) − pool(q). In thống kê TB |NEW|/câu, đóng góp riêng từng nguồn,
          trùng nhau giữa các nguồn, phân bố hạng trong nguồn của doc NEW. Xếp NEW theo RRF k60
          giữa các nguồn chứa nó (rank trong từng nguồn) -> new_cands.json {qid: [doc_id]} +
          new_cands_meta.json [{doc_id, n_src, sources: [[tên, rank], ...], rrf}, ...].
2. subs   submission_ceil2: relevant_docs = toàn pool (thứ tự cache_pool_rr) ∪ NEW (RRF);
            relevant_chunks = y hệt --cx130 (doc_id chunk còn thiếu thì nối cuối).
          submission_newonly30: relevant_docs = --n doc đầu của NEW theo RRF; relevant_chunks = [].
3. zipcheck  kiểm zip đã đóng gói: đúng 1 entry gốc, <= --max-zip, in sha256 + kích thước.

Dùng:
  python scripts/candgen14.py stats --pool cache_pool_rr.json --pool2 cache_pool2_rr.json \
      --hyde dense_hyde_top200.json --reason dense_reason_top200.json --dense dense_full_top400.json \
      --bm25 cache_vi_k300.json --out-dir D:\\r2ai\\data\\sub\\diag18
  python scripts/candgen14.py subs --pool cache_pool_rr.json --new diag18\\new_cands.json \
      --cx130 diag17\\submission_cx130cov.json --out-dir D:\\r2ai\\data\\sub\\diag18
  python scripts/candgen14.py zipcheck --zip diag18\\submission_ceil2.zip
"""
import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

K_RRF = 60
N_NEWONLY = 30
ZIP_LIMIT = 95_000_000
N_QUERIES = 1200
RANK_BANDS = ((1, 50, "1-50"), (51, 100, "51-100"), (101, 200, "101-200"),
              (201, 300, "201-300"), (301, 10 ** 9, "301+"))
SRC_ORDER = ("pool2", "hyde", "reason", "dense", "bm25")


def load_cache(path) -> dict:
    """Đọc cache {qid: [[doc_id, idx, score], ...]} -> {str(qid): hits}."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {str(q): v for q, v in data.items()}


def hit_ids(hits: list, limit: int | None = None) -> list:
    """doc_id (int) theo thứ tự hits, bỏ trùng lần xuất hiện đầu; limit=None lấy hết."""
    out, seen = [], set()
    for h in hits[:limit] if limit else hits:
        d = int(h[0])
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def source_ranks(hits: list, limit: int | None = None) -> dict:
    """{doc_id: rank 1-based} trong nguồn (bỏ trùng, rank = vị trí trong danh sách đã cắt)."""
    out = {}
    for i, h in enumerate(hits[:limit] if limit else hits):
        out.setdefault(int(h[0]), i + 1)
    return out


def new_candidates(pool_hits: list, sources: dict) -> dict:
    """NEW của một câu = (hợp nguồn) − pool.

    sources = {tên: hits}; trả {doc_id: {tên: rank trong nguồn}}.
    """
    pool = set(hit_ids(pool_hits))
    out: dict = {}
    for name, hits in sources.items():
        for d, r in source_ranks(hits).items():
            if d in pool:
                continue
            out.setdefault(d, {})[name] = r
    return out


def rrf_score(ranks: dict, k: int = K_RRF) -> float:
    """RRF: tổng 1/(k + rank) trên các nguồn chứa doc."""
    return sum(1.0 / (k + r) for r in ranks.values())


def order_new(new_map: dict, k: int = K_RRF, src_order: tuple = SRC_ORDER) -> list:
    """Xếp NEW theo RRF giảm dần; đồng điểm: hạng tốt nhất tăng dần rồi doc_id tăng dần.

    Trả [(doc_id, rrf, [(tên nguồn theo src_order, rank), ...]), ...].
    """
    def key(item):
        d, ranks = item
        best = min(ranks.values())
        names = [s for s in src_order if s in ranks] + sorted(
            s for s in ranks if s not in src_order)
        return (-rrf_score(ranks, k), best, d, tuple(names))

    rows = []
    for d, ranks in sorted(new_map.items(), key=key):
        names = [s for s in src_order if s in ranks] + sorted(s for s in ranks if s not in src_order)
        rows.append((d, round(rrf_score(ranks, k), 8), [[s, ranks[s]] for s in names]))
    return rows


def rank_band(r: int) -> str:
    for a, b, name in RANK_BANDS:
        if a <= r <= b:
            return name
    return RANK_BANDS[-1][2]


def stats_new(per_q_new: dict, src_names: tuple = SRC_ORDER) -> dict:
    """Thống kê NEW trên mọi câu.

    per_q_new = {qid: {doc_id: {src: rank}}}; trả dict TB |NEW|/câu, đóng góp riêng/từng nguồn,
    trùng nhau từng cặp nguồn, phân bố hạng trong nguồn.
    """
    qs = list(per_q_new)
    n = max(len(qs), 1)
    sizes = [len(per_q_new[q]) for q in qs]
    uniq = {s: 0 for s in src_names}
    present = {s: 0 for s in src_names}
    pair = {f"{a}|{b}": 0 for i, a in enumerate(src_names) for b in src_names[i + 1:]}
    bands = {s: {name: 0 for _, _, name in RANK_BANDS} for s in src_names}
    rank_vals = {s: [] for s in src_names}
    for q in qs:
        for _, ranks in per_q_new[q].items():
            for s in src_names:
                if s in ranks:
                    present[s] += 1
                    bands[s][rank_band(ranks[s])] += 1
                    rank_vals[s].append(ranks[s])
            if len(ranks) == 1:
                uniq[next(iter(ranks))] += 1
            for key in pair:
                a, b = key.split("|")
                if a in ranks and b in ranks:
                    pair[key] += 1
    srcs = {}
    for s in src_names:
        rv = rank_vals[s]
        srcs[s] = {"TB_new_co_mat": present[s] / n, "TB_new_rieng": uniq[s] / n,
                   "TB_hang": sum(rv) / len(rv) if rv else None,
                   "min_hang": min(rv) if rv else None, "max_hang": max(rv) if rv else None,
                   "phan_bo_hang": {name: bands[s][name] / n for _, _, name in RANK_BANDS}}
    return {"so_cau": len(qs), "TB_new_cau": sum(sizes) / n, "min_new": min(sizes, default=0),
            "max_new": max(sizes, default=0), "tong_new": sum(sizes),
            "nguon": srcs, "trung_cap_nguon": {key: pair[key] / n for key in pair}}


def per_query_new(pool: dict, sources: dict) -> dict:
    """{str(qid): {doc_id: {src: rank}}} cho mọi câu của pool."""
    srcs = [{str(q): v for q, v in d.items()} for d in sources]
    out = {}
    for q, hits in pool.items():
        out[q] = new_candidates(hits, {name: s.get(q, []) for name, s in zip(SRC_ORDER, srcs)})
    return out


def write_new_cands(per_q_new: dict, rows_q: dict, out_dir) -> tuple:
    """Ghi new_cands.json {qid: [doc_id]} + new_cands_meta.json; trả 2 đường dẫn."""
    out = Path(out_dir)
    cands = {q: [d for d, _, _ in rows] for q, rows in rows_q.items()}
    meta = {q: [{"doc_id": d, "n_src": len(rs), "sources": rs, "rrf": sc} for d, sc, rs in rows]
            for q, rows in rows_q.items()}
    p1 = out / "new_cands.json"
    p2 = out / "new_cands_meta.json"
    p1.write_text(json.dumps(cands, ensure_ascii=False), encoding="utf-8")
    p2.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return p1, p2


def build_ceil2(cx_sub: list, pool: dict, new_rows: dict) -> tuple:
    """relevant_docs = toàn pool (thứ tự cache) ∪ NEW (RRF); chunks = y hệt cx130.

    new_rows = {str(qid): [(doc_id, ...), ...]}; trả (out, stats).
    """
    out, added = [], []
    for it in cx_sub:
        q = str(it["id"])
        docs = hit_ids(pool.get(q, []))
        seen = set(docs)
        for d, _, _ in new_rows.get(q, []):
            if d not in seen:
                seen.add(d)
                docs.append(d)
        chunks = list(it["relevant_chunks"])
        n_add = 0
        for ch in chunks:
            d = int(ch["doc_id"])
            if d not in seen:
                seen.add(d)
                docs.append(d)
                n_add += 1
        added.append(n_add)
        out.append({"id": it["id"], "relevant_docs": docs, "relevant_chunks": chunks})
    n = max(len(out), 1)
    stats = {"TB_doc_cau": sum(len(s["relevant_docs"]) for s in out) / n,
             "TB_chunk_cau": sum(len(s["relevant_chunks"]) for s in out) / n,
             "TB_doc_them_cho_chunk": sum(added) / n, "tong_doc_them_cho_chunk": sum(added)}
    return out, stats


def build_newonly30(new_rows: dict, n_docs: int = N_NEWONLY) -> list:
    """relevant_docs = n_docs doc đầu của NEW theo RRF; relevant_chunks = [] (mọi câu, theo thứ tự id)."""
    out = []
    for q in sorted(new_rows, key=lambda x: int(x)):
        docs = [d for d, _, _ in new_rows[q][:n_docs]]
        out.append({"id": int(q), "relevant_docs": docs, "relevant_chunks": []})
    return out


def zipcheck(zip_path, max_zip: int = ZIP_LIMIT) -> dict:
    """Kiểm zip: đúng 1 entry ở gốc, tên submission.json, <= max_zip, nén đọc lại được."""
    zp = Path(zip_path)
    size = zp.stat().st_size
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        one = len(names) == 1 and "/" not in names[0] and "\\" not in names[0]
        ok_name = names == ["submission.json"] if names else False
        bad = z.testzip()
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    rep = {"zip": str(zp), "bytes": size, "duoi_gioi_han": size <= max_zip,
           "so_entry": len(names), "entry_goc": one, "ten_dung": ok_name,
           "doc_lai": bad is None, "sha256": sha, "sha256_rut_gon": sha[:16],
           "ok": bool(one and ok_name and bad is None and size <= max_zip)}
    return rep


def cmd_stats(a) -> int:
    pool = load_cache(a.pool)
    srcs = {"pool2": load_cache(a.pool2)}
    if a.hyde:
        srcs["hyde"] = load_cache(a.hyde)
    if a.reason:
        srcs["reason"] = load_cache(a.reason)
    if a.dense:
        srcs["dense"] = {q: v[: a.dense_k] for q, v in load_cache(a.dense).items()}
    if a.bm25:
        srcs["bm25"] = {q: v[: a.bm25_k] for q, v in load_cache(a.bm25).items()}
    names = tuple(s for s in SRC_ORDER if s in srcs)
    per_q = {}
    for q, hits in pool.items():
        per_q[q] = new_candidates(hits, {name: srcs[name].get(q, []) for name in names})
    rows_q = {q: order_new(new_map, a.k, names) for q, new_map in per_q.items()}
    rep = stats_new(per_q, names)
    rep["nguon_khong_co"] = [s for s in SRC_ORDER if s not in srcs]
    rep["k_rrf"] = a.k
    p1, p2 = write_new_cands(per_q, rows_q, a.out_dir)
    rep["new_cands"] = str(p1)
    rep["new_cands_meta"] = str(p2)
    txt = json.dumps(rep, ensure_ascii=False)
    print(txt, flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")
    return 0


def cmd_subs(a) -> int:
    pool = load_cache(a.pool)
    cx = json.loads(Path(a.cx130).read_text(encoding="utf-8"))
    cands = {str(q): [int(d) for d in v]
             for q, v in json.loads(Path(a.new).read_text(encoding="utf-8")).items()}
    new_rows = {q: [(d, 0.0, []) for d in ds] for q, ds in cands.items()}
    ceil2, st2 = build_ceil2(cx, pool, new_rows)
    only = build_newonly30(new_rows, a.n)
    ids2 = {s["id"] for s in ceil2}
    ids3 = {s["id"] for s in only}
    st2["so_cau"] = len(ceil2)
    st2["id_khop_cx130"] = ids2 == {it["id"] for it in cx}
    p1 = Path(a.out_dir) / "submission_ceil2.json"
    p2 = Path(a.out_dir) / "submission_newonly30.json"
    p1.write_text(json.dumps(ceil2, ensure_ascii=False), encoding="utf-8")
    p2.write_text(json.dumps(only, ensure_ascii=False), encoding="utf-8")
    nd3 = [len(s["relevant_docs"]) for s in only]
    rep = {"ceil2": st2, "newonly30": {"so_cau": len(only), "id_khop_cx130": ids3 == ids2,
                                       "TB_doc_cau": sum(nd3) / max(len(nd3), 1),
                                       "max_doc": max(nd3, default=0), "chunk_cau": 0.0},
           "new_cands": a.new, "ghi": [str(p1), str(p2)]}
    txt = json.dumps(rep, ensure_ascii=False)
    print(txt, flush=True)
    if a.report:
        Path(a.report).write_text(txt, encoding="utf-8")
    return 0


def cmd_zipcheck(a) -> int:
    rep = zipcheck(a.zip, a.max_zip)
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    if a.report:
        Path(a.report).write_text(json.dumps(rep, ensure_ascii=False), encoding="utf-8")
    return 0 if rep["ok"] else 3


def main() -> None:
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    st = sp.add_parser("stats")
    st.add_argument("--pool", required=True, help="cache_pool_rr.json (pool hiện tại)")
    st.add_argument("--pool2", required=True, help="cache_pool2_rr.json")
    st.add_argument("--hyde", default="", help="dense_hyde_top200.json")
    st.add_argument("--reason", default="", help="dense_reason_top200.json")
    st.add_argument("--dense", default="", help="dense_full_top400.json (lấy --dense-k đầu)")
    st.add_argument("--bm25", default="", help="cache_vi_k300.json (lấy --bm25-k đầu)")
    st.add_argument("--dense-k", type=int, default=300)
    st.add_argument("--bm25-k", type=int, default=300)
    st.add_argument("--k", type=int, default=K_RRF)
    st.add_argument("--out-dir", required=True)
    st.add_argument("--report", default="")
    sb = sp.add_parser("subs")
    sb.add_argument("--pool", required=True)
    sb.add_argument("--new", required=True, help="new_cands.json của stats")
    sb.add_argument("--cx130", required=True, help="submission_cx130cov.json (nguồn chunk)")
    sb.add_argument("--out-dir", required=True)
    sb.add_argument("--n", type=int, default=N_NEWONLY)
    sb.add_argument("--report", default="")
    zc = sp.add_parser("zipcheck")
    zc.add_argument("--zip", required=True)
    zc.add_argument("--max-zip", type=int, default=ZIP_LIMIT)
    zc.add_argument("--report", default="")
    a = ap.parse_args()
    sys.exit({"stats": cmd_stats, "subs": cmd_subs, "zipcheck": cmd_zipcheck}[a.cmd](a))


if __name__ == "__main__":
    main()
