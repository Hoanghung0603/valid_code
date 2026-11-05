"""Gộp chunk ứng viên cũ (chunk_dw_u900 top-120) với chunk mới (docwin mở rộng D_q) — chạy CPU b660.

Chunk mới = đầu ra chunks_from_docwin.py --units 900 trên docwin_ext (doc_rank = hạng trong
relevant_docs của submission_split); chunk cũ = docwin\\chunk_dw_u900.json.zst (doc_rank = hạng
trong top-100 cache_pool_rr). Hai nguồn dùng chung định dạng 5 cột
[doc_id, doc_rank, chunk_idx, logit, chunk_text] nhưng doc_rank THEO HAI QUY ƯỚC KHÁC NHAU.

1. merge   chunk cũ ∪ chunk mới, mỗi câu sắp logit giảm dần (đồng logit: doc_rank, chunk_idx
           tăng), KHÔNG cắt top-120 -> <out> (mặc định chunk_dw_u900_ext.json.zst).
2. verify  mọi chunk_text là NGUYÊN VĂN của text doc (--docs-new với chunk mới, --docs-old nếu
           muốn kiểm luôn chunk cũ); số sai phải = 0.
3. stats   TB chunk mới/câu, TB doc mới có chunk/câu, TB chunk/câu sau gộp; phân bố logit
           (p25/p50/p75) chunk mới vs cũ: toàn thể + theo cùng hạng trong danh sách logit của
           từng loại.
4. impact  chọn lại theo s = z(logit chunk) + wd·z(điểm fused của doc) như chunk_docfuse.py,
           k_q = số chunk của bài cũ (--ref-chunks); đếm số chunk MỚI lọt vào top-k_q.

Vì cands dùng toàn bộ danh sách nộp (để doc_rank = hạng nộp), bước GPU chấm thêm cặp (q, doc)
top-100 bị export ở câu khác; --dq-cands + --pool lọc chunk mới về đúng D_q (docngoài top-100
bge của câu) trước khi gộp/thống kê; --new-out ghi chunk mới đã lọc.

Dùng:
  python scripts/dwx_merge.py --old docwin\\chunk_dw_u900.json.zst --new-dir docwin_ext \
      --docs-new docwin_ext\\docwin_ext.part000.jsonl.zst ... --docs-old chunkdocs100\\... \
      --fused diag10\\cache_rrf3d2.json --ref-chunks diag12\\submission_split.json \
      --out docwin_ext\\chunk_dw_u900_ext.json.zst --report docwin_ext\\dwx_merge_report.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from chunk_docfuse import pick_q
from chunks_from_docwin import load_json, verify_verbatim
from rerank_run import load_docs

RANKS = [1, 15, 40, 60, 80, 100, 120]


def pct(vals: list, p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * (len(s) - 1) + 0.5))]


def dist3(vals: list) -> dict:
    return {"p25": pct(vals, 0.25), "p50": pct(vals, 0.5), "p75": pct(vals, 0.75)}


def sort_rows(rows: list) -> list:
    """Sắp logit giảm dần; đồng logit: doc_rank, chunk_idx tăng dần (như select_top)."""
    return sorted(rows, key=lambda r: (-r[3], r[1], r[2]))


def merge_q(old_rows: list, new_rows: list) -> list:
    return sort_rows(list(old_rows) + list(new_rows))


def find_new_file(new_dir: Path, stem: str = "chunk_dw_u900") -> Path:
    """File chunk mới trong new_dir: <stem>.json hoặc <stem>.json.zst (chunks_from_docwin.write_out)."""
    for name in (f"{stem}.json.zst", f"{stem}.json"):
        p = new_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"không thấy {stem}.json[.zst] trong {new_dir}")


def dq_docs_per_q(cands: dict, pool: dict, top_n: int = 100) -> dict:
    """{qid: set(str(doc_id))} — doc trong danh sách nộp KHÔNG thuộc top-N pool (bge).

    Chunk mới chỉ được tính cho các doc này (doc top-100 đã có docwin/chunk cũ; vì cands dùng
    toàn bộ danh sách nộp để doc_rank đúng, bước GPU có chấm thêm cặp top-100 bị export ở câu
    khác — phải loại khỏi chunk mới ở đây).
    """
    po = {str(k): v for k, v in pool.items()}
    out = {}
    for q, hits in cands.items():
        top = {str(h[0]) for h in (po.get(str(q), []) or [])[:top_n]}
        out[str(q)] = {str(item[0]) for item in hits if str(item[0]) not in top}
    return out


def filter_new_rows(new_by_q: dict, dq_sets: dict) -> tuple[dict, int, int]:
    """Giữ chunk mới có doc_id ∈ D_q của câu; trả (mới đã lọc, số dòng giữ, số dòng bỏ)."""
    kept, drop = {}, 0
    for q, rows in new_by_q.items():
        ok = dq_sets.get(str(q), set())
        rows_q = [r for r in rows if str(r[0]) in ok]
        drop += len(rows) - len(rows_q)
        if rows_q:
            kept[q] = rows_q
    return kept, sum(len(r) for r in kept.values()), drop


def write_zst_stream(out: Path, data: dict) -> Path:
    """Ghi {qid: rows} thành JSON nén zstd theo dòng chảy (không dựng chuỗi JSON khổng lồ)."""
    import zstandard
    with open(out, "wb") as f:
        with zstandard.ZstdCompressor(level=6).stream_writer(f) as w:
            w.write(b"{")
            first = True
            for q, rows in data.items():
                if not first:
                    w.write(b",")
                first = False
                w.write(json.dumps(str(q), ensure_ascii=False).encode("utf-8"))
                w.write(b":")
                w.write(json.dumps(rows, ensure_ascii=False).encode("utf-8"))
            w.write(b"}")
    return out


def rankwise_logits(rows_by_q: dict, ranks: list = RANKS) -> dict:
    """{rank: [logit]} — logit tại hạng rank (1-based, theo logit giảm dần) trên các câu đủ dòng."""
    out = {rk: [] for rk in ranks}
    for rows in rows_by_q.values():
        s = sort_rows(rows)
        for rk in ranks:
            if len(s) >= rk:
                out[rk].append(s[rk - 1][3])
    return out


def new_doc_stats(new_by_q: dict, nq: int | None = None) -> tuple[float, float]:
    """(TB chunk mới/câu, TB doc mới có chunk/câu); nq = số câu mẫu (mặc định số câu có chunk mới)."""
    nq = nq or max(len(new_by_q), 1)
    nc = sum(len(r) for r in new_by_q.values()) / nq
    nd = sum(len({str(x[0]) for x in r}) for r in new_by_q.values()) / nq
    return nc, nd


def impact_stats(merged_by_q: dict, new_by_q: dict, fused: dict, ref_chunks: list, wd: float) -> dict:
    """Chọn theo s = zc + wd·zd (chunk_docfuse.pick_q), k_q = số chunk của bài cũ.

    Trả thống kê số chunk MỚI lọt top-k_q (doc_id của chunk mới = doc D_q, không trùng doc cũ).
    """
    fu = {str(k): v for k, v in fused.items()}
    rr = {str(q): v for q, v in merged_by_q.items()}
    nw = {str(q): v for q, v in new_by_q.items()}
    newsel, ks, hits_q = [], [], 0
    for it in ref_chunks:
        q = str(it["id"])
        k = len(it["relevant_chunks"])
        ks.append(k)
        selected = pick_q(rr.get(q, []), fu.get(q, []), k, wd)
        new_docs = {str(r[0]) for r in nw.get(q, [])}
        n_new = sum(1 for c in selected if str(c["doc_id"]) in new_docs)
        newsel.append(n_new)
        if n_new:
            hits_q += 1
    nq = max(len(ref_chunks), 1)
    return {
        "wd": wd,
        "TB k_q (số chunk bài cũ)": sum(ks) / nq if ks else 0.0,
        "TB chunk mới lọt top-k/câu": sum(newsel) / nq,
        "TB chunk mới lọt / k_q": (sum(newsel) / sum(ks)) if ks and sum(ks) else 0.0,
        "câu có chunk mới lọt": hits_q,
        "số câu": len(ref_chunks),
        "phân bố số mới lọt": {"min": min(newsel) if newsel else 0, "max": max(newsel) if newsel else 0,
                               "TB": sum(newsel) / nq},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="chunk_dw_u900.json.zst cũ")
    ap.add_argument("--new-dir", required=True, help="thư mục chứa chunk_dw_u900.json[.zst] mới")
    ap.add_argument("--docs-new", required=True, nargs="+", help="JSONL.zst text doc D_q (docwin_ext)")
    ap.add_argument("--docs-old", nargs="*", default=[], help="JSONL.zst text doc cũ (chunkdocs100) để verify")
    ap.add_argument("--fused", required=True, help="cache_rrf3d2.json {qid: [[doc_id, idx, fused], ...]}")
    ap.add_argument("--ref-chunks", required=True, help="bài cũ (k_q = số relevant_chunks mỗi câu)")
    ap.add_argument("--out", required=True, help="file gộp (ghi .json.zst)")
    ap.add_argument("--report", default="", help="JSON thống kê (mặc định <out>.report.json)")
    ap.add_argument("--dq-cands", default="", help="cands_docwin_ext.json để lọc chunk mới về D_q")
    ap.add_argument("--pool", default="", help="cache_pool_rr.json (cần khi --dq-cands)")
    ap.add_argument("--top-n", type=int, default=100, help="top-N pool tính là 'đã có docwin'")
    ap.add_argument("--new-out", default="", help="ghi chunk mới đã lọc D_q ra JSON.zst")
    ap.add_argument("--wd", type=float, default=4.0)
    ap.add_argument("--ranks", type=int, nargs="+", default=RANKS)
    ap.add_argument("--top-chunks", type=int, default=0, help=">0 thì cắt mỗi câu sau khi gộp (0 = giữ hết)")
    a = ap.parse_args()
    t0 = time.time()

    old = {str(q): v for q, v in load_json(a.old).items()}
    new_path = find_new_file(Path(a.new_dir))
    new = {str(q): v for q, v in load_json(str(new_path)).items()}
    print(f"chunk cũ {len(old)} câu ({Path(a.old).name}), chunk mới {len(new)} câu ({new_path.name}), "
          f"{time.time() - t0:.0f}s", flush=True)

    raw_rows = sum(len(r) for r in new.values())
    raw_docs = sum(len({str(x[0]) for x in r}) for r in new.values())
    n_keep = n_drop = 0
    if a.dq_cands and a.pool:
        cands = json.loads(Path(a.dq_cands).read_text(encoding="utf-8"))
        pool = json.loads(Path(a.pool).read_text(encoding="utf-8"))
        new, n_keep, n_drop = filter_new_rows(new, dq_docs_per_q(cands, pool, a.top_n))
        print(f"lọc D_q: giữ {n_keep} dòng chunk mới, bỏ {n_drop} dòng doc top-{a.top_n} bị chấm trùng", flush=True)
        if a.new_out:
            p = write_zst_stream(Path(a.new_out), new)
            print(f"ghi chunk mới D_q {p} ({p.stat().st_size / 1e6:.0f}MB)", flush=True)

    qids = [q for q in old] + [q for q in new if q not in old]
    merged = {}
    for q in qids:
        rows = merge_q(old.get(q, []), new.get(q, []))
        merged[q] = rows[: a.top_chunks] if a.top_chunks > 0 else rows

    out_path = Path(a.out)
    if not out_path.name.endswith(".zst"):
        out_path = out_path.with_name(out_path.name + ("" if out_path.name.endswith(".json") else ".json") + ".zst")
    write_zst_stream(out_path, merged)
    nc = [len(r) for r in merged.values()]
    nq = max(len(merged), 1)
    print(f"ghi {out_path} ({out_path.stat().st_size / 1e6:.0f}MB): {len(merged)} câu, "
          f"TB {sum(nc) / nq:.1f} dòng/câu ({time.time() - t0:.0f}s)", flush=True)

    docs_new = load_docs(a.docs_new)
    bad_new = verify_verbatim({q: rows for q, rows in new.items()}, docs_new)
    bad_old = -1
    if a.docs_old:
        old_docs_wanted = {r[0] for rows in old.values() for r in rows}
        docs_old = load_docs(a.docs_old, wanted=old_docs_wanted)
        bad_old = verify_verbatim(old, docs_old)
    print(f"nguyên văn: chunk mới sai {bad_new}/{sum(len(r) for r in new.values())}, "
          f"chunk cũ sai {bad_old}", flush=True)

    fused = json.loads(Path(a.fused).read_text(encoding="utf-8"))
    ref = json.loads(Path(a.ref_chunks).read_text(encoding="utf-8"))
    nc_new, nd_new = new_doc_stats(new, nq)
    all_new = [r[3] for rows in new.values() for r in rows]
    all_old = [r[3] for rows in old.values() for r in rows]
    rank_new = rankwise_logits(new, a.ranks)
    rank_old = rankwise_logits(old, a.ranks)
    rep = {
        "chunk mới/câu TB": nc_new,
        "doc mới có chunk/câu TB": nd_new,
        "chunk mới thô/câu TB (trước lọc D_q)": raw_rows / nq,
        "doc mới thô/câu TB (trước lọc D_q)": raw_docs / nq,
        "dòng mới bị loại (doc top-100)": n_drop,
        "dòng/câu sau gộp TB": sum(nc) / nq,
        "câu": len(merged),
        "nguyên văn mới sai": bad_new,
        "nguyên văn cũ sai": bad_old,
        "logit chunk mới (toàn thể)": dist3(all_new),
        "logit chunk cũ (toàn thể)": dist3(all_old),
        "logit theo hạng (mới)": {str(k): dist3(v) for k, v in rank_new.items()},
        "logit theo hạng (cũ)": {str(k): dist3(v) for k, v in rank_old.items()},
        "impact": impact_stats(merged, new, fused, ref, a.wd),
    }
    report_path = Path(a.report) if a.report else out_path.with_name(out_path.name.replace(".json.zst", "") + ".report.json")
    report_path.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(rep, ensure_ascii=False), flush=True)
    print(f"ghi báo cáo {report_path}; tổng {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
