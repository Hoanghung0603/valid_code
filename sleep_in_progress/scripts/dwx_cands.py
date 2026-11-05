"""Tính tập D_q (doc của submission_split nằm NGOÀI top-100 bge) cho docwin mở rộng — chạy CPU b660.

Với mỗi câu q trong submission_split.json:
  - top100 = 100 doc đầu của cache_pool_rr.json[q] (thứ tự bge, như docwin cũ);
  - D_q = doc trong relevant_docs (giữ thứ tự = hạng trong danh sách nộp) KHÔNG thuộc top100.
idx (chỉ số BigIndex để export text) lấy từ cache_pool_rr (mọi hạng), thiếu thì cache_fhyde.

Đầu ra trong --out-dir:
  cache_docwin_ext.json  {qid: [[doc_id, idx, 0.0], ...]} chỉ D_q (theo hạng nộp) — nạp cho
                         export_docs.py --cache --n (bản ghi "cache" giả, điểm không dùng tới).
  cands_docwin_ext.json  {qid: [[doc_id, idx], ...]} TOÀN BỘ relevant_docs theo hạng nộp — nạp cho
                         docwin_run.py và chunks_from_docwin.py. doc_rank = vị trí 1-based trong
                         danh sách này = hạng trong submission_split; doc ngoài D_q không có text/
                         wins nên bị bỏ qua nhưng hạng của doc D_q vẫn đúng.
  dwx_cands_report.json  thống kê: số cặp (q, doc), doc duy nhất, phân theo dải hạng, doc thiếu idx.

Dùng:
  python scripts/dwx_cands.py --split D:\\r2ai\\data\\sub\\diag12\\submission_split.json \
      --pool D:\\r2ai\\data\\sub\\cache_pool_rr.json --fhyde D:\\r2ai\\data\\sub\\diag12\\cache_fhyde.json \
      --out-dir D:\\r2ai\\data\\sub\\docwin_ext
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BANDS = ((1, 30, "1-30"), (31, 60, "31-60"), (61, 100, "61-100"), (101, 10 ** 9, "101+"))


def load_json(path: str) -> dict | list:
    p = Path(path)
    if p.name.endswith(".zst"):
        import zstandard
        with open(p, "rb") as f:
            raw = zstandard.ZstdDecompressor().stream_reader(f).read()
        return json.loads(raw)
    return json.loads(p.read_text(encoding="utf-8"))


def norm_id(d):
    return int(d) if str(d).lstrip("-").isdigit() else d


def band_of(rank: int) -> str:
    for a, b, k in BANDS:
        if a <= rank <= b:
            return k
    return BANDS[-1][2]


def top_ids(hits: list, n: int) -> set:
    return {str(h[0]) for h in (hits or [])[:n]}


def idx_lookup(pool_hits: list, fhyde_hits: list) -> dict:
    """{str(doc_id): idx} từ cache_pool_rr trước, thiếu thì cache_fhyde."""
    out = {}
    for h in (pool_hits or []) + (fhyde_hits or []):
        d = str(h[0])
        if len(h) > 1 and h[1] is not None:
            out.setdefault(d, int(h[1]))
    return out


def q_d_docs(sub_docs: list, top100: set, idxs: dict,
             rank_band: tuple[int, int] | None = None) -> tuple[list, list]:
    """D_q của một câu: [(hạng_nộp, doc_id, có_idx)] giữ thứ tự; trả (dq, missing).

    missing = [[hạng_nộp, doc_id]] là doc D_q không có idx trong pool/fhyde (không export được).
    rank_band (lo, hi) nếu có thì chỉ giữ doc có hạng trong [lo, hi] (ưu tiên hạng thấp khi chạy
    theo đợt); doc ngoài band bị loại khỏi tập chạy nhưng hạng nộp của doc được giữ không đổi.
    """
    out, missing = [], []
    for r, d in enumerate(sub_docs, 1):
        s = str(d)
        if s in top100:
            continue
        if rank_band and not (rank_band[0] <= r <= rank_band[1]):
            continue
        did = norm_id(d)
        ok = s in idxs
        out.append((r, did, ok))
        if not ok:
            missing.append([r, did])
    return out, missing


def build(split: list, pool: dict, fhyde: dict, top_n: int = 100,
          rank_band: tuple[int, int] | None = None) -> tuple[dict, dict, dict]:
    """Trả (cands_full, cache_ext, report). cands_full giữ đủ thứ tự nộp; cache_ext chỉ D_q có idx."""
    po = {str(k): v for k, v in pool.items()}
    fu = {str(k): v for k, v in fhyde.items()}
    cands_full, cache_ext = {}, {}
    pairs = pairs_ok = 0
    uniq: set = set()
    band_pairs = {k: 0 for _, _, k in BANDS}
    miss = []
    dq_lens, sub_lens = [], []
    for it in split:
        q = str(it["id"])
        sub_docs = it["relevant_docs"]
        pool_hits = po.get(q, [])
        t100 = top_ids(pool_hits, top_n)
        idxs = idx_lookup(pool_hits, fu.get(q, []))
        dq, missing = q_d_docs(sub_docs, t100, idxs, rank_band)
        miss.extend([[q] + m for m in missing])
        dq_lens.append(len(dq))
        sub_lens.append(len(sub_docs))
        pairs += len(dq)
        pairs_ok += sum(1 for _, _, ok in dq if ok)
        for r, d, _ok in dq:
            uniq.add(d)
            band_pairs[band_of(r)] += 1
        cands_full[q] = [[norm_id(d), idxs.get(str(d))] for d in sub_docs]
        cache_ext[q] = [[d, idxs[str(d)], 0.0] for _, d, ok in dq if ok]
    nq = max(len(split), 1)
    report = {
        "câu": len(split),
        "cặp (q, doc) D_q": pairs,
        "cặp export được (có idx)": pairs_ok,
        "TB D_q/câu": pairs / nq,
        "D_q min/max": [min(dq_lens) if dq_lens else 0, max(dq_lens) if dq_lens else 0],
        "doc duy nhất": len(uniq),
        "TB relevant_docs/câu": sum(sub_lens) / nq if sub_lens else 0.0,
        "cặp theo dải hạng nộp": band_pairs,
        "doc thiếu idx": miss,
        "top_n": top_n,
        "rank_band": list(rank_band) if rank_band else None,
    }
    return cands_full, cache_ext, report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, help="submission_split.json [{id, relevant_docs, relevant_chunks}]")
    ap.add_argument("--pool", required=True, help="cache_pool_rr.json {qid: [[doc_id, idx, logit], ...]}")
    ap.add_argument("--fhyde", required=True, help="cache_fhyde.json (idx dự phòng)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=100, help="top-N của pool tính là 'đã có docwin'")
    ap.add_argument("--rank-band", type=int, nargs=2, default=None, metavar=("LO", "HI"),
                    help="chỉ export doc có hạng nộp trong [LO, HI] (cands vẫn giữ đủ thứ tự)")
    a = ap.parse_args()

    split = load_json(a.split)
    pool = load_json(a.pool)
    fhyde = load_json(a.fhyde)
    band = tuple(a.rank_band) if a.rank_band else None
    if band and band[0] > band[1]:
        ap.error("rank_band: LO phải <= HI")
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cands_full, cache_ext, report = build(split, pool, fhyde, a.top_n, band)
    (out_dir / "cands_docwin_ext.json").write_text(json.dumps(cands_full), encoding="utf-8")
    (out_dir / "cache_docwin_ext.json").write_text(json.dumps(cache_ext), encoding="utf-8")
    (out_dir / "dwx_cands_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "doc thiếu idx"}, ensure_ascii=False), flush=True)
    print(f"doc thiếu idx: {len(report['doc thiếu idx'])}", flush=True)
    print(f"ghi cands_docwin_ext.json + cache_docwin_ext.json + dwx_cands_report.json vào {out_dir}", flush=True)


if __name__ == "__main__":
    main()
