"""Cào các trang 39.net (subdomain baby/care/heart/fk/nk/...) còn thiếu trong bộ ZH 235k: bản desktop dính captcha,
nhưng UA điện thoại → https://m.39.net/<sub>/a_<id>.html không captcha. Giữ doc_id gốc. Chạy lại được.
    python scripts/fetch_39m.py --out data/cache/zh39m --conc 6
"""
import argparse, asyncio, glob, os, re, sys, time
import pandas as pd
from curl_cffi.requests import AsyncSession
sys.path.insert(0, ".")
from src.extractors import extract  # noqa: E402
UA = "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1"


def murl(u):
    m = re.match(r"https?://(\w+)\.39\.net/a/\d+/(\d+)\.html", u)
    return f"https://m.39.net/{m.group(1)}/a_{m.group(2)}.html" if m else None


async def worker(q, sess, out, st):
    while True:
        it = await q.get()
        if it is None:
            q.task_done(); return
        did, url, mu = it
        for att in range(3):
            try:
                r = await sess.get(mu, headers={"User-Agent": UA}, timeout=30, allow_redirects=True)
                if r.status_code == 200 and "verify" not in str(r.url):
                    e = extract(r.content, mu, r.headers.get("content-type"))
                    if len(e.get("text") or "") >= 50:
                        out.append(dict(doc_id=did, source="39m", lang="zh", url=url, title=e.get("title") or "", text=e["text"], fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"))); st["ok"] += 1
                    else: st["empty"] += 1
                    break
                st[f"h{r.status_code}"] = st.get(f"h{r.status_code}", 0) + 1
                if r.status_code == 404: break
                await asyncio.sleep(3 * (att + 1))
            except Exception:
                st["err"] += 1; await asyncio.sleep(3 * (att + 1))
        await asyncio.sleep(0.2); q.task_done()


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/cache/zh39m"); ap.add_argument("--conc", type=int, default=6); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    M = pd.read_parquet("data/cache/tmp/zh_missing.parquet"); M = M[M.host.str.endswith(".39.net")]
    done = set()
    for f in glob.glob(f"{a.out}/part-*.parquet"): done |= set(pd.read_parquet(f, columns=["doc_id"]).doc_id)
    todo = [(str(i), u, murl(u)) for i, u in zip(M.id, M.url) if str(i) not in done and murl(u)]
    if a.limit: todo = todo[:a.limit]
    print("cần cào", len(todo), "(39.net thiếu", len(M), ")", flush=True); st = {"ok": 0, "empty": 0, "err": 0}; t0 = time.time()
    async with AsyncSession(impersonate="safari_ios") as sess:
        for b in range(0, len(todo), 500):
            q, out = asyncio.Queue(), []
            for it in todo[b:b + 500]: q.put_nowait(it)
            for _ in range(a.conc): q.put_nowait(None)
            await asyncio.gather(*[worker(q, sess, out, st) for _ in range(a.conc)])
            if out: pd.DataFrame(out).to_parquet(f"{a.out}/part-{int(time.time())}.parquet", index=False)
            print(f"[{time.time()-t0:.0f}s] {min(b+500, len(todo))}/{len(todo)} {st}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
