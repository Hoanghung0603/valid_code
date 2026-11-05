"""Cào lại các URL VI còn thiếu ở những host đã cào được phần lớn (data/cache/tmp/missing_vi.parquet: id, url, host).
Bỏ trang video laodong (không có chữ). Trích xuất bằng src.extractors.extract (giống crawler chính). Chạy lại được.
    python scripts/refetch_missing.py --out data/cache/refetch --conc 6
"""
import argparse, asyncio, glob, os, sys, time
import pandas as pd
from curl_cffi.requests import AsyncSession
sys.path.insert(0, ".")
from src.extractors import extract  # noqa: E402


async def worker(q, sess, out, stats):
    while True:
        it = await q.get()
        if it is None:
            q.task_done(); return
        did, url = it
        for att in range(3):
            try:
                r = await sess.get(url, timeout=40, impersonate="chrome", allow_redirects=True)
                if r.status_code == 200:
                    e = extract(r.content, url, r.headers.get("content-type"))
                    if len(e.get("text") or "") >= 200 and e.get("lang") == "vi":
                        out.append(dict(doc_id=did, source="refetch", lang="vi", url=url, title=e.get("title") or "", text=e["text"], fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S")))
                        stats["ok"] += 1
                    else:
                        stats["empty"] += 1
                    break
                if r.status_code in (404, 410):
                    stats["404"] += 1; break
                stats[f"http{r.status_code}"] = stats.get(f"http{r.status_code}", 0) + 1; await asyncio.sleep(5 * (att + 1))
            except Exception:
                stats["err"] += 1; await asyncio.sleep(5 * (att + 1))
        await asyncio.sleep(0.3); q.task_done()


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/cache/refetch"); ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    M = pd.read_parquet("data/cache/tmp/missing_vi.parquet"); M = M[~M.url.str.contains("laodong.vn/video")]
    done = set()
    for f in glob.glob(f"{a.out}/part-*.parquet"):
        done |= set(pd.read_parquet(f, columns=["doc_id"]).doc_id)
    todo = [(str(i), u) for i, u in zip(M.id, M.url) if str(i) not in done]
    todo.sort(key=lambda x: hash(x[1]) % 1000)   # trộn host ⇒ không dồn 1 host
    if a.limit: todo = todo[:a.limit]
    print("cần cào", len(todo), M.host.value_counts().to_dict(), flush=True)
    stats = {"ok": 0, "empty": 0, "404": 0, "err": 0}; t0 = time.time()
    async with AsyncSession() as sess:
        for b in range(0, len(todo), 500):
            q, out = asyncio.Queue(), []
            for it in todo[b:b + 500]: q.put_nowait(it)
            for _ in range(a.conc): q.put_nowait(None)
            await asyncio.gather(*[worker(q, sess, out, stats) for _ in range(a.conc)])
            if out:
                pd.DataFrame(out).to_parquet(f"{a.out}/part-{int(time.time())}.parquet", index=False)
            print(f"[{time.time()-t0:.0f}s] {min(b+500, len(todo))}/{len(todo)} {stats}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
