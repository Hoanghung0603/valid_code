"""Ensemble hai cache điểm {qid: [[doc_id, idx, score], ...]} (bge + Qwen) thành cache xếp hạng mới.

Ứng viên mỗi câu = giao 100 doc đầu của --a (tập ứng viên của Qwen) với doc của --b; doc chỉ có ở
một bên bị bỏ. Doc thiếu điểm (None) ở một bên nhận điểm thấp nhất của bên đó trong câu (tính trên
tập ứng viên); câu không có điểm nào thì nhận 0.

--mode rrf: điểm = wa/(k + hạng_a) + wb/(k + hạng_b); hạng 1-based theo score giảm dần từng bên
  sau khi điền điểm (đồng điểm giữ thứ tự file của bên đó).
--mode z: z-score từng bên trong câu (z = (s − TB)/σ, σ = 0 → z = 0), điểm = wa·z_a + wb·z_b.

Đầu ra {qid: [[doc_id, idx, score], ...]} (idx lấy từ --a), score tròn 6 chữ số, sắp giảm dần;
đồng điểm giữ thứ tự --a (bge). In trùng top-30 TB với top-30 của từng mô hình.

Dùng:
  python scripts/ensemble_cache.py --a data/sub/cache_pool_rr.json --b data/sub/qwen06_top100.json \
    --mode rrf --k 60 --out data/sub/qwen06/cache_ens_rrf.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def fill_min(scores: dict, order: list) -> dict:
    """Điền điểm None bằng điểm thấp nhất (không None) trong câu; không có thì 0.0."""
    vals = [scores[d] for d in order if scores[d] is not None]
    fill = min(vals) if vals else 0.0
    return {d: (fill if scores[d] is None else scores[d]) for d in order}


def rank_map(order: list, scores: dict) -> dict:
    """doc_id -> hạng 1-based theo score giảm dần; đồng điểm giữ thứ tự order."""
    srt = sorted(range(len(order)), key=lambda i: (-scores[order[i]], i))
    return {order[i]: r + 1 for r, i in enumerate(srt)}


def z_map(order: list, scores: dict) -> dict:
    """doc_id -> z-score trong câu; σ = 0 (hoặc câu rỗng) thì z = 0."""
    vals = [scores[d] for d in order]
    n = len(vals)
    if n == 0:
        return {}
    mean = sum(vals) / n
    std = (sum((v - mean) ** 2 for v in vals) / n) ** 0.5
    if std == 0:
        return {d: 0.0 for d in order}
    return {d: (v - mean) / std for d, v in zip(order, vals)}


def top30_overlap(out_rows: list, model_order: list) -> float:
    """Số doc/30 trùng giữa top-30 đầu ra và top-30 của một mô hình."""
    return len({r[0] for r in out_rows[:30]} & set(model_order[:30])) / 30.0


def ens_query(rows_a: list, rows_b: list, mode: str, k: float = 60,
              wa: float = 1.0, wb: float = 1.0) -> tuple[list, list, list]:
    """(dòng [doc_id, idx, score] đã sắp, hạng_a, hạng_b) của một câu.

    hạng_a/hạng_b = thứ tự doc theo score của từng bên (tập ứng viên chung), dùng cho thống kê.
    """
    pos_a = {r[0]: i for i, r in enumerate(rows_a)}
    order_a = [r[0] for r in rows_a]
    idx_a = {r[0]: r[1] for r in rows_a}
    raw_a = {r[0]: r[2] for r in rows_a}
    order_b = [r[0] for r in rows_b]
    raw_b = {r[0]: r[2] for r in rows_b}
    common = [d for d in order_a if d in raw_b]
    if not common:
        return [], [], []
    order_b_c = [d for d in order_b if d in set(common)]
    sc_a = fill_min({d: raw_a[d] for d in common}, common)
    sc_b = fill_min({d: raw_b[d] for d in common}, order_b_c)
    model_a = sorted(common, key=lambda d: (-sc_a[d], pos_a[d]))
    pos_b = {d: i for i, d in enumerate(order_b_c)}
    model_b = sorted(order_b_c, key=lambda d: (-sc_b[d], pos_b[d]))
    if mode == "rrf":
        ra, rb = rank_map(common, sc_a), rank_map(order_b_c, sc_b)
        ens = {d: wa / (k + ra[d]) + wb / (k + rb[d]) for d in common}
    else:
        za, zb = z_map(common, sc_a), z_map(order_b_c, sc_b)
        ens = {d: wa * za[d] + wb * zb[d] for d in common}
    srt = sorted(common, key=lambda d: (-ens[d], pos_a[d]))
    rows = [[d, idx_a[d], round(ens[d], 6)] for d in srt]
    return rows, model_a, model_b


def ensemble_cache(cache_a: dict, cache_b: dict, mode: str, k: float = 60, wa: float = 1.0,
                   wb: float = 1.0, a_top: int = 100) -> tuple[dict, dict]:
    """Trả (cache đầu ra, thống kê trùng top-30 {'trùng top-30 a': ..., 'trùng top-30 b': ...}).

    Câu không có trong --b bị bỏ. --a chỉ xét a_top doc đầu mỗi câu.
    """
    out, ov_a, ov_b = {}, [], []
    for q, rows_a in cache_a.items():
        rows_b = cache_b.get(q)
        if rows_b is None:
            continue
        rows, model_a, model_b = ens_query(rows_a[:a_top], rows_b, mode, k, wa, wb)
        out[q] = rows
        if rows:
            ov_a.append(top30_overlap(rows, model_a))
            ov_b.append(top30_overlap(rows, model_b))
    n = max(len(ov_a), 1)
    stats = {"trùng top-30 a": sum(ov_a) / n, "trùng top-30 b": sum(ov_b) / n}
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Ensemble hai cache điểm (rrf hoặc z-score)")
    ap.add_argument("--a", required=True, help="cache bge; chỉ xét --a-top doc đầu mỗi câu")
    ap.add_argument("--b", required=True, help="cache Qwen (cùng định dạng {qid: [[doc_id, idx, score], ...]})")
    ap.add_argument("--mode", required=True, choices=["rrf", "z"])
    ap.add_argument("--k", type=float, default=60, help="hằng số RRF")
    ap.add_argument("--wa", type=float, default=1.0, help="trọng số --a")
    ap.add_argument("--wb", type=float, default=1.0, help="trọng số --b")
    ap.add_argument("--a-top", type=int, default=100, help="số doc đầu của --a (mặc định 100)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cache_a = json.loads(Path(a.a).read_text(encoding="utf-8"))
    cache_b = json.loads(Path(a.b).read_text(encoding="utf-8"))
    out, stats = ensemble_cache(cache_a, cache_b, a.mode, a.k, a.wa, a.wb, a.a_top)
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    n = max(len(out), 1)
    print(f"ghi {a.out}: {len(out)} câu, mode={a.mode} k={a.k} wa={a.wa} wb={a.wb}; "
          f"trùng top-30 a TB {stats['trùng top-30 a']:.4f} ({30 * stats['trùng top-30 a']:.1f}/30); "
          f"trùng top-30 b TB {stats['trùng top-30 b']:.4f} ({30 * stats['trùng top-30 b']:.1f}/30)", flush=True)


if __name__ == "__main__":
    main()
