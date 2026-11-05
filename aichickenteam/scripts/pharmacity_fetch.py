"""Cào lại 5.3k trang Pharmacity (BTC 19/10: pmc-ecm-healthblog.beta.pharmacity.io/<slug>/ chết → www.pharmacity.vn/<slug>.htm; GIỮ doc_id gốc).
Trích xuất: div.pmc-content-blog-detail, mỗi khối (p/h2/h3/li/tr/blockquote) một dòng (tránh dính chữ). Chạy lại được (bỏ id đã có).
    python scripts/pharmacity_fetch.py --out data/cache/pharmacity --conc 4
Ra: data/cache/pharmacity/part-*.parquet (doc_id, source, lang, url, title, text, fetched_at) cùng schema data/docs.
"""
import argparse, asyncio, glob, os, re, time
import pandas as pd
from curl_cffi.requests import AsyncSession
from selectolax.parser import HTMLParser



_BLOCK = re.compile(r"<(/?)(h[1-6]|p|li|ul|ol|tr|td|th|div|br|blockquote|table|figure|figcaption|section)\b[^>]*>", re.I)


def extract(html):
    """Nội dung trong #pmc-content-blog-detail; văn bản trần + <strong> làm tiêu đề ⇒ ranh giới thẻ khối = xuống dòng, thẻ khác = khoảng trắng."""
    t = HTMLParser(html)
    h1 = t.css_first("h1"); ti = t.css_first("title")
    title = (h1.text(strip=True) if h1 else "") or (ti.text(strip=True) if ti else "")
    root = t.css_first("#pmc-content-blog-detail")
    if root is None:
        return title, ""
    for bad in root.css("script, style, noscript, iframe, svg, button, img"):
        bad.decompose()
    inner = _BLOCK.sub("\n", root.html)
    inner = re.sub(r"</?strong[^>]*>", "\n", inner, flags=re.I)   # <strong> là tiêu đề con (ngay sau là chữ dính liền)
    txt = HTMLParser(f"<div>{inner}</div>").css_first("div").text(separator=" ")
    lines = [re.sub(r"\s+", " ", x).strip() for x in txt.split("\n")]
    return title, "\n".join(x for x in lines if x)


async def worker(name, q, sess, out, stats):
    while True:
        item = await q.get()
        if item is None:
            q.task_done(); return
        did, url, slug = item
        new = f"https://www.pharmacity.vn/{slug}.htm"
        for att in range(3):
            try:
                r = await sess.get(new, timeout=40, impersonate="chrome")
                if r.status_code == 200:
                    title, text = extract(r.text)
                    if len(text) >= 200:
                        out.append(dict(doc_id=did, source="pharmacity-new", lang="vi", url=url, title=title, text=text, fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S")))
                        stats["ok"] += 1
                    else:
                        stats["empty"] += 1
                    break
                if r.status_code == 404:
                    stats["404"] += 1; break
                stats[f"http{r.status_code}"] = stats.get(f"http{r.status_code}", 0) + 1
                await asyncio.sleep(5 * (att + 1))
            except Exception:
                stats["err"] += 1; await asyncio.sleep(5 * (att + 1))
        await asyncio.sleep(0.3)
        q.task_done()


async def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="data/cache/pharmacity"); ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    L = pd.read_parquet("data/ViBioMIR/links_corpus.parquet")
    L = L[L.url.str.contains("pmc-ecm-healthblog.beta.pharmacity.io", na=False)].copy()
    L["slug"] = L.url.str.replace(r"^https?://[^/]+/", "", regex=True).str.strip("/")
    done = set()
    for f in glob.glob(f"{a.out}/part-*.parquet"):
        done |= set(pd.read_parquet(f, columns=["doc_id"]).doc_id)
    todo = [(str(i), u, s) for i, u, s in zip(L.id, L.url, L.slug) if str(i) not in done and s]
    if a.limit: todo = todo[:a.limit]
    print("tổng", len(L), "đã có", len(done), "cần cào", len(todo), flush=True)
    stats = {"ok": 0, "empty": 0, "404": 0, "err": 0}; t0 = time.time()
    async with AsyncSession() as sess:
        for b in range(0, len(todo), 500):
            q, out = asyncio.Queue(), []
            for it in todo[b:b + 500]: q.put_nowait(it)
            for _ in range(a.conc): q.put_nowait(None)
            await asyncio.gather(*[worker(i, q, sess, out, stats) for i in range(a.conc)])
            if out:
                pd.DataFrame(out).to_parquet(f"{a.out}/part-{int(time.time())}.parquet", index=False)
            print(f"[{time.time()-t0:.0f}s] {b+500}/{len(todo)} {stats}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
