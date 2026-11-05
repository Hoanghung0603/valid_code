"""Gộp logit rerank cũ + mới: pool mở rộng (rerank_pool2.json) + cache xếp hạng mới (cache_pool2_rr.json).

--old (rerank_pool.json: {qid: [[doc_id, idx, logit], ...]}) giữ NGUYÊN giá trị (logit lẫn idx lẫn thứ tự);
thêm hit mới từ --new (rerank_run.py) theo thứ tự rerank, bỏ doc đã có (nguồn cũ thắng).
--out: JSON gộp cùng định dạng rerank_pool.json. --out-cache: --out sắp theo fuse.sort_by_score (logit
giảm dần, None cuối) — cùng định dạng cache_pool_rr.json.

Kiểm tra với --ref (cache_pool2_rr phải thừa hưởng): đủ câu, không trùng doc, mọi doc của ref còn nguyên
logit + idx; lệch thì thoát mã 1. In thống kê quanh --thr (TB pool/câu, TB số doc logit > thr trước/sau,
TB số doc MỚI logit > thr).

Ví dụ:
  python scripts/merge_pool.py --old data/rerank_pool.json --new data/rerank_new2.json \
      --out data/rerank_pool2.json --out-cache data/cache_pool2_rr.json --ref data/cache_pool_rr.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from fuse import sort_by_score


def merge_pool(old: dict, new: dict) -> dict:
    """{qid: [[doc_id, idx, logit], ...]}: hit cũ theo thứ tự cũ + hit mới (bỏ doc đã có, bỏ trùng)."""
    out = {}
    for q, hits in old.items():
        seen = {h[0] for h in hits}
        merged = [list(h) for h in hits]
        for h in new.get(q, []):
            if h[0] in seen:
                continue
            seen.add(h[0])
            merged.append(list(h))
        out[q] = merged
    for q, hits in new.items():
        if q not in out:
            seen = set()
            merged = []
            for h in hits:
                if h[0] in seen:
                    continue
                seen.add(h[0])
                merged.append(list(h))
            out[q] = merged
    return out


def check_ref(merged: dict, ref: dict) -> list:
    """Danh sách lỗi: lệch bộ câu, trùng doc, doc ref thiếu/idx lệch/logit lệch trong merged."""
    errs = []
    if set(merged) != set(ref):
        errs.append(f"lệch bộ câu: merged {len(merged)} vs ref {len(ref)}")
    for q, hits in ref.items():
        if q not in merged:
            errs.append(f"câu {q}: thiếu trong merged")
            continue
        got = {}
        for h in merged[q]:
            if h[0] in got:
                errs.append(f"câu {q}: doc {h[0]} trùng trong merged")
            got[h[0]] = h
        for h in hits:
            g = got.get(h[0])
            if g is None:
                errs.append(f"câu {q}: doc {h[0]} của ref thiếu trong merged")
            elif g[2] != h[2] or g[1] != h[1]:
                errs.append(f"câu {q}: doc {h[0]} lệch logit/idx ({g[1:]}) vs ref ({h[1:]})")
    return errs


def pool_stats(pool: dict, old: dict, thr: float) -> dict:
    """TB pool/câu, TB số doc logit > thr, TB số doc MỚI (ngoài old) logit > thr; None không tính."""
    n = max(len(pool), 1)
    n_pool = n_gt = n_new_gt = 0
    for q, hits in pool.items():
        have_old = {h[0] for h in old.get(q, [])}
        for h in hits:
            n_pool += 1
            if h[2] is not None and h[2] > thr:
                n_gt += 1
                if h[0] not in have_old:
                    n_new_gt += 1
    return {"pool": n_pool / n, "gt": n_gt / n, "new_gt": n_new_gt / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, help="rerank_pool.json")
    ap.add_argument("--new", required=True, help="JSON {qid: [[doc_id, idx, logit], ...]} từ rerank_run.py")
    ap.add_argument("--out", required=True, help="rerank_pool2.json (gộp, giữ nguyên hit cũ)")
    ap.add_argument("--out-cache", required=True, help="cache_pool2_rr.json (logit giảm dần)")
    ap.add_argument("--ref", required=True, help="cache_pool_rr.json để đối chiếu logit cũ")
    ap.add_argument("--thr", type=float, default=-2.6)
    a = ap.parse_args()

    old = json.loads(Path(a.old).read_text(encoding="utf-8"))
    new = json.loads(Path(a.new).read_text(encoding="utf-8"))
    ref = json.loads(Path(a.ref).read_text(encoding="utf-8"))
    merged = merge_pool(old, new)
    errs = check_ref(merged, ref)

    Path(a.out).write_text(json.dumps(merged), encoding="utf-8")
    cache = {q: sort_by_score(hits) for q, hits in merged.items()}
    Path(a.out_cache).write_text(json.dumps(cache), encoding="utf-8")

    s_before = pool_stats(old, old, a.thr)
    s_after = pool_stats(merged, old, a.thr)
    n_new = sum(len(hits) - len(old.get(q, [])) for q, hits in merged.items())
    print(f"ghi {a.out} + {a.out_cache}: {len(merged)} câu, {n_new} doc mới", flush=True)
    print(f"pool/câu: TB {s_before['pool']:.1f} → {s_after['pool']:.1f}", flush=True)
    print(f"doc logit > {a.thr}/câu: TB {s_before['gt']:.1f} → {s_after['gt']:.1f}", flush=True)
    print(f"doc MỚI logit > {a.thr}/câu: TB {s_after['new_gt']:.1f}", flush=True)
    if errs:
        print(f"LỖI kiểm ref ({len(errs)}): {errs[:5]}", flush=True)
        sys.exit(1)
    print("kiểm ref OK: đủ câu, không trùng doc, logit/idx cũ giữ nguyên", flush=True)


if __name__ == "__main__":
    main()
