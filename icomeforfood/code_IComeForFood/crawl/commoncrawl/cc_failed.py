"""Which failed / empty non-Long Chau URLs of the crawl DB are archived in Common Crawl (index only, newest 200 record)?

Big hosts: one prefix query per snapshot (paged). Small hosts: exact query per URL, newest snapshot first,
stop at the first status-200 hit, so the kept record is always the newest one.

    python -I cc_failed.py <status.duckdb> <out_dir>
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import httpx

sys.stdout.reconfigure(encoding="utf-8")
db, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
N_SNAP, PREFIX_MIN = 24, 100
FL = "url,timestamp,filename,offset,length,mime,status"


def key(u: str) -> str:
    p = urlparse(u.strip())
    host = p.netloc.lower().removeprefix("www.")
    return host + p.path.rstrip("/") + (("?" + p.query) if p.query else "")


rows = duckdb.connect(db, read_only=True).sql(
    "select id, url, status from crawl_status where status not in ('ok', 'pending') "
    "and url not like '%nhathuoclongchau.com.vn%'").fetchall()
want = {key(u): (i, u, s) for i, u, s in rows}
by_host = defaultdict(list)
for k, (i, u, s) in want.items():
    by_host[urlparse(u).netloc.lower()].append(k)
print(f"{len(want)} failed URLs on {len(by_host)} hosts", flush=True)

http = httpx.Client(timeout=300, headers={"User-Agent": "r2ai-research-cc-index-check"})


def get(url: str, params: dict):
    for attempt in range(6):
        try:
            r = http.get(url, params=params)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text
        except Exception:
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f"index query failed: {url} {params}")


cols = [c["id"] for c in http.get("https://index.commoncrawl.org/collinfo.json").json()][:N_SNAP]  # newest first
hits_f = out / "cc_hits.json"
done_f = out / "cc_done.txt"
hits = json.loads(hits_f.read_text(encoding="utf-8")) if hits_f.exists() else {}
done = set(done_f.read_text().split()) if done_f.exists() else set()


def keep(x: dict) -> None:
    k = key(x["url"])
    if k in want and x.get("status") == "200" and (k not in hits or x["timestamp"] > hits[k]["timestamp"]):
        hits[k] = x


def save(tag: str) -> None:
    hits_f.write_text(json.dumps(hits, ensure_ascii=False), encoding="utf-8")
    with open(done_f, "a") as f:
        f.write(tag + "\n")
    done.add(tag)


for host, keys in sorted(by_host.items(), key=lambda kv: -len(kv[1])):
    if len(keys) >= PREFIX_MIN:
        for c in cols:
            tag = f"{host}|{c}"
            if tag in done:
                continue
            api = f"https://index.commoncrawl.org/{c}-index"
            q = {"url": f"{host}/*", "output": "json", "filter": "status:200", "fl": FL}
            meta = get(api, {**q, "showNumPages": "true"})
            pages = json.loads(meta)["pages"] if meta else 0
            for p in range(pages):
                for line in (get(api, {**q, "page": p}) or "").splitlines():
                    if line.startswith("{"):
                        keep(json.loads(line))
                time.sleep(1)
            save(tag)
            print(f"{host} {c} pages {pages}  hits so far {sum(k in hits for k in keys)}/{len(keys)}", flush=True)
    else:
        for k in keys:
            u = want[k][1]
            if f"url|{u}" in done:
                continue
            for c in cols:
                text = get(f"https://index.commoncrawl.org/{c}-index",
                           {"url": u, "output": "json", "filter": "status:200", "fl": FL})
                found = [json.loads(l) for l in (text or "").splitlines() if l.startswith("{")]
                for x in found:
                    keep(x)
                time.sleep(1)
                if k in hits:
                    break
            save(f"url|{u}")
        print(f"{host} exact  hits {sum(k in hits for k in keys)}/{len(keys)}", flush=True)

print("== summary (failed URLs with a status-200 CC record, newest snapshot kept)")
for host, keys in sorted(by_host.items(), key=lambda kv: -len(kv[1])):
    ts = sorted(hits[k]["timestamp"][:8] for k in keys if k in hits)
    print(f"{host:45s} {len(ts):6d}/{len(keys):<6d} newest-record dates {ts[0] if ts else '-'}..{ts[-1] if ts else '-'}")
print("ALL_DONE")
