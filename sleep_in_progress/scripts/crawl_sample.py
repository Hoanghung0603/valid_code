"""Crawl thử một mẫu URL phân tầng theo domain để đo tốc độ, tỷ lệ lỗi và chất lượng nội dung.

Chạy: python scripts/crawl_sample.py --per-host 10
Kết quả: data/sample_crawl.jsonl (mỗi dòng 1 URL) + bảng tổng hợp in ra màn hình.
"""
import argparse
import asyncio
import json
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import aiohttp
import pandas as pd
import trafilatura

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0 Safari/537.36"
ROOT = Path(__file__).resolve().parent.parent


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


async def fetch(session, sem_global, host_locks, delay, row):
    host = row["host"]
    rec = {"id": int(row["id"]), "url": row["url"], "host": host}
    async with host_locks[host]:  # tối đa 1 request đồng thời mỗi host
        async with sem_global:
            t0 = time.perf_counter()
            try:
                async with session.get(row["url"], allow_redirects=True) as r:
                    body = await r.read()
                    rec.update(status=r.status, final_url=str(r.url), bytes=len(body))
            except Exception as e:  # timeout, DNS, SSL...
                rec.update(status=None, error=type(e).__name__, bytes=0)
                body = b""
            rec["fetch_s"] = round(time.perf_counter() - t0, 2)
        if body and rec.get("status") == 200:
            # trafilatura tự xử lý encoding (UTF-8/GBK...) khi nhận bytes
            text = await asyncio.to_thread(
                trafilatura.extract, body, include_comments=False, include_tables=True, favor_recall=True
            )
            rec["text_len"] = len(text or "")
            rec["text"] = text or ""
        await asyncio.sleep(delay)  # lịch sự với server
    return rec


async def main(args):
    corpus = pd.read_parquet(ROOT / "data/links_corpus.parquet")
    corpus["host"] = corpus.url.map(host_of)
    sample = corpus.groupby("host", group_keys=False).apply(
        lambda g: g.sample(min(len(g), args.per_host), random_state=42)
    )
    print(f"Sample: {len(sample)} URL từ {sample.host.nunique()} host")

    host_locks = defaultdict(asyncio.Lock)
    sem = asyncio.Semaphore(args.concurrency)
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    conn = aiohttp.TCPConnector(limit=args.concurrency, ssl=False)
    t0 = time.perf_counter()
    async with aiohttp.ClientSession(headers={"User-Agent": UA}, timeout=timeout, connector=conn) as s:
        tasks = [fetch(s, sem, host_locks, args.delay, r) for r in sample.to_dict("records")]
        results = []
        for i, fut in enumerate(asyncio.as_completed(tasks), 1):
            results.append(await fut)
            if i % 100 == 0:
                print(f"  {i}/{len(tasks)} ({time.perf_counter() - t0:.0f}s)")
    wall = time.perf_counter() - t0

    out = ROOT / "data/sample_crawl.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    df = pd.DataFrame(results)
    df["ok"] = (df.status == 200) & (df.get("text_len", 0).fillna(0) >= 200)
    summ = df.groupby("host").agg(
        n=("id", "size"), ok_rate=("ok", "mean"),
        fetch_s=("fetch_s", "median"), text_len=("text_len", "median"),
    ).sort_values("n", ascending=False)
    pd.set_option("display.width", 200)
    print(summ.to_string())
    print(f"\nTổng: {len(df)} URL, OK {df.ok.mean():.1%}, wall {wall:.0f}s, "
          f"status: {df.status.value_counts(dropna=False).to_dict()}")
    print(f"Lỗi: {df.get('error', pd.Series(dtype=str)).value_counts().to_dict()}")
    print(f"Đã lưu {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--per-host", type=int, default=10)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--delay", type=float, default=1.0)
    p.add_argument("--timeout", type=float, default=20.0)
    asyncio.run(main(p.parse_args()))
