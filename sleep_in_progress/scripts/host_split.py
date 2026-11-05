"""Chia top-N doc mỗi câu của cache theo host URL: H (y tế chuyên dụng) vs N (báo/tổng hợp, còn lại).

Host = hostname URL (bỏ www., chữ thường). H nếu host chứa bất kỳ chuỗi nào trong --h-keys.
Ghi hai cache con {qid: [[doc_id, idx, score], ...]} GIỮ NGUYÊN THỨ TỰ: --out-h, --out-n;
câu không có doc nhóm nào thì list rỗng (evaluate.py validate chấp nhận).
Báo cáo --report: top 40 host theo số lần trong top-N (đếm, nhóm), số câu có ≥1 doc H / ≥1 doc N
/ cả hai, TB doc/câu mỗi nhóm (trên mọi câu và trên câu có ≥1 doc).

Dùng:
  python scripts/host_split.py --cache cache_qwen8b_sorted.json --links links_corpus.parquet \
    --out-h cache_q8b_H.json --out-n cache_q8b_N.json --report host_report.txt --top 30
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

sys.stdout.reconfigure(encoding="utf-8")

DEFAULT_H_KEYS = [
    "vinmec", "hellobacsi", "medlatec", "suckhoe", "longchau", "tamanhhospital", "benhvien",
    "hospital", "bachmai", "youmed", "doctor", "medic", "thuoc", "ykhoa", "pharmacity",
    "isofhcare", "zhongyi", "wujue",
]


def host_of(url: str) -> str:
    """Hostname chữ thường, bỏ www. (vd 'https://www.Vinmec.com/a' -> 'vinmec.com')."""
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def is_h(host: str, h_keys: list[str]) -> bool:
    """True nếu host chứa bất kỳ chuỗi nào trong h_keys (chữ thường)."""
    return any(k.lower() in host for k in h_keys)


def split_cache(cache: dict, id2url: dict, h_keys: list[str], top: int = 30) -> tuple[dict, dict, dict]:
    """(cache_h, cache_n, stats) — hai cache con giữ thứ tự; stats: host counter + thống kê nhóm."""
    h_cache, n_cache = {}, {}
    cnt = Counter()
    n_q_h = n_q_n = n_q_both = 0
    sum_h = sum_n = 0
    for q, rows in cache.items():
        h_rows, n_rows = [], []
        for row in rows[:top]:
            host = host_of(id2url.get(row[0], ""))
            grp = "H" if is_h(host, h_keys) else "N"
            cnt[(host, grp)] += 1
            (h_rows if grp == "H" else n_rows).append(row)
        h_cache[q], n_cache[q] = h_rows, n_rows
        sum_h += len(h_rows)
        sum_n += len(n_rows)
        n_q_h += bool(h_rows)
        n_q_n += bool(n_rows)
        n_q_both += bool(h_rows) and bool(n_rows)
    n = len(cache) or 1
    stats = {
        "top": top,
        "h_keys": list(h_keys),
        "hosts": [{"host": h, "nhóm": g, "số lần": c} for (h, g), c in
                  sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0][0]))],
        "câu": len(cache),
        "câu có ≥1 H": n_q_h,
        "câu có ≥1 N": n_q_n,
        "câu có cả H và N": n_q_both,
        "TB doc H/câu": round(sum_h / n, 4),
        "TB doc N/câu": round(sum_n / n, 4),
        "TB doc H/câu (câu có ≥1 H)": round(sum_h / n_q_h, 4) if n_q_h else 0.0,
        "TB doc N/câu (câu có ≥1 N)": round(sum_n / n_q_n, 4) if n_q_n else 0.0,
    }
    return h_cache, n_cache, stats


def format_report(stats: dict, top_hosts: int = 40) -> str:
    """Bảng top host (đếm, nhóm) + thống kê nhóm."""
    lines = [f"top-{stats['top']} mỗi câu, {stats['câu']} câu",
             f"h_keys ({len(stats['h_keys'])}): {', '.join(stats['h_keys'])}",
             "", f"top {top_hosts} host theo số lần xuất hiện:", "số lần  nhóm  host"]
    for item in stats["hosts"][:top_hosts]:
        lines.append(f"{item['số lần']:>6}  {item['nhóm']:>4}  {item['host']}")
    lines += ["", f"số câu có ≥1 doc H: {stats['câu có ≥1 H']}",
              f"số câu có ≥1 doc N: {stats['câu có ≥1 N']}",
              f"số câu có cả H và N: {stats['câu có cả H và N']}",
              f"TB doc H/câu: {stats['TB doc H/câu']} (câu có ≥1 H: {stats['TB doc H/câu (câu có ≥1 H)']})",
              f"TB doc N/câu: {stats['TB doc N/câu']} (câu có ≥1 N: {stats['TB doc N/câu (câu có ≥1 N)']})"]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="JSON {qid: [[doc_id, idx, score], ...]}")
    ap.add_argument("--links", required=True, help="links_corpus.parquet (id, url)")
    ap.add_argument("--out-h", required=True)
    ap.add_argument("--out-n", required=True)
    ap.add_argument("--report", default="", help="ghi bảng host + thống kê nhóm")
    ap.add_argument("--top", type=int, default=30)
    ap.add_argument("--h-keys", nargs="+", default=DEFAULT_H_KEYS)
    ap.add_argument("--top-hosts", type=int, default=40)
    a = ap.parse_args()

    import pandas as pd

    cache = json.loads(Path(a.cache).read_text(encoding="utf-8"))
    wanted = {row[0] for rows in cache.values() for row in rows[: a.top]}
    df = pd.read_parquet(a.links, columns=["id", "url"])
    df = df[df["id"].isin(wanted)]
    id2url = dict(zip(df["id"], df["url"]))
    h_cache, n_cache, stats = split_cache(cache, id2url, a.h_keys, a.top)
    Path(a.out_h).write_text(json.dumps(h_cache, ensure_ascii=False), encoding="utf-8")
    Path(a.out_n).write_text(json.dumps(n_cache, ensure_ascii=False), encoding="utf-8")
    rep = format_report(stats, a.top_hosts)
    if a.report:
        Path(a.report).write_text(rep + "\n", encoding="utf-8")
    print(f"ghi {a.out_h} ({sum(len(v) for v in h_cache.values())} dòng) + {a.out_n} "
          f"({sum(len(v) for v in n_cache.values())} dòng)", flush=True)
    print(rep, flush=True)


if __name__ == "__main__":
    main()
