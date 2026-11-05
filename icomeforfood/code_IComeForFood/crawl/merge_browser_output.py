"""Hợp nhất đầu ra của `crawl/crawl_browser.py` (hoặc `crawl/crawl_home.py`) vào pipeline chính.

Notebook 01 và bước trích chỉ đọc `crawl/html/shard_*.jsonl.zst` và bảng `crawl_status`, nên shard
của bộ thu bằng trình duyệt phải được đánh số lại vào thư mục đó, còn `status.csv` phải upsert
vào DuckDB. Script chỉ thêm, không sửa shard đã có và không đụng id đã `ok` trong DB.

    python crawl/merge_browser_output.py --out D:/r2ai_data/crawl_browser \
        --html-dir D:/r2ai_data/crawl/html --status-db status.duckdb

Chạy lại nhiều lần được: `merge_state.json` trong thư mục `--out` ghi shard nào đã chép sang tên nào.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import duckdb


def read_status(path: Path) -> dict[int, dict]:
    """Dòng cuối cho mỗi id; dòng bị cắt khi mất điện (thiếu status) bị bỏ."""
    last: dict[int, dict] = {}
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("id") or "").isdigit() and r.get("status"):
                last[int(r["id"])] = r
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path, help="thư mục đầu ra của bộ thu")
    ap.add_argument("--html-dir", required=True, type=Path, help="crawl/html của pipeline chính")
    ap.add_argument("--status-db", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    status = read_status(args.out / "status.csv")
    done = [r for r in status.values() if r["status"] in ("ok", "non_html", "empty")]
    print(f"status.csv: {len(status):,} id | có nội dung: {len(done):,}")

    state_path = args.out / "merge_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {"shards": {}}
    existing = sorted(args.html_dir.glob("shard_*.jsonl.zst"))
    next_idx = max((int(m.group(1)) for p in existing if (m := re.search(r"shard_(\d+)", p.name))), default=-1) + 1
    print(f"{args.html_dir}: {len(existing)} shard, shard mới bắt đầu từ {next_idx:05d}")

    con = duckdb.connect(str(args.status_db))
    try:
        done_ids = {r[0] for r in con.execute(
            "SELECT id FROM crawl_status WHERE status IN ('ok', 'non_html')").fetchall()}
        # giữ đúng shard đã ghi trong status; id đã ok trong DB thì bỏ qua
        to_merge = [r for r in done if int(r["id"]) not in done_ids]
        print(f"bỏ qua {len(done) - len(to_merge):,} id đã ok trong DB; cần hợp nhất {len(to_merge):,}")

        shard_map: dict[str, int] = state["shards"]
        copied = 0
        for name in sorted({r["shard_file"] for r in to_merge if r.get("shard_file")}):
            if name in shard_map:
                continue
            src = args.out / "html" / name
            if not src.exists():
                print(f"thiếu shard {name}; các id của nó chưa được hợp nhất")
                continue
            if args.dry_run:
                shard_map[name] = -1
                continue
            target = args.html_dir / f"shard_{next_idx:05d}.jsonl.zst"
            if target.exists():
                raise SystemExit(f"từ chối ghi đè {target}")
            shutil.copy2(src, target)
            shard_map[name] = next_idx
            next_idx += 1
            copied += 1
            print(f"chép {name} -> {target.name}")
        if not args.dry_run:
            (args.out / "merge_state.json").write_text(
                json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8")

        rows = []
        for r in to_merge:
            idx = shard_map.get(r.get("shard_file") or "")
            if r["status"] in ("ok", "non_html", "empty") and idx is None:
                continue  # shard chưa có thì để lần sau
            rows.append((
                r.get("final_url") or r["url"], r["status"],
                int(r["http_code"]) if (r.get("http_code") or "").isdigit() else None,
                r.get("content_type"), int(r["n_bytes"]) if (r.get("n_bytes") or "").isdigit() else 0,
                int(r["attempts"]) if (r.get("attempts") or "").isdigit() else 1,
                r.get("fetched_at") or datetime.now(timezone.utc).isoformat(),
                idx, r.get("sha1_raw"), r.get("error_msg"), int(r["id"]),
            ))
        if args.dry_run:
            print(f"[dry-run] sẽ cập nhật {len(rows):,} dòng, chép {len({v for v in shard_map.values() if v >= 0})} shard")
            return
        if rows:
            con.execute("BEGIN TRANSACTION")
            try:
                con.executemany("""
                    UPDATE crawl_status
                    SET final_url=?, status=?, http_code=?, content_type=?, n_bytes=?, attempts=?,
                        fetched_at=?, shard=?, sha1_raw=?, error_msg=?
                    WHERE id=?
                """, rows)
                con.execute("COMMIT")
            except Exception:
                con.execute("ROLLBACK")
                raise
            con.execute("CHECKPOINT")
        print(f"đã cập nhật {len(rows):,} dòng status; chép {copied} shard")
        print(con.execute("""
            SELECT status, COUNT(*) AS n FROM crawl_status
            WHERE lower(url) LIKE '%nhathuoclongchau.com.vn%' GROUP BY 1 ORDER BY n DESC
        """).df().to_string())
    finally:
        con.close()


if __name__ == "__main__":
    main()
