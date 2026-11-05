import glob
import json
import os

from zstandard import ZstdDecompressor

BASE = r"D:\r2ai\data\crawl\async_recovery_20261009"
QUEUE = os.path.join(BASE, "queue")
DONE = os.path.join(BASE, "state", "done_ids.tsv")


def is_vi(host):
    h = (host or "").lower()
    return h.endswith(".vn") or h == "vinmec.com"


status = {}
cols = {}
with open(DONE, "r", encoding="utf-8", errors="replace") as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.rstrip("\n").split("\t")
        cols[len(parts)] = cols.get(len(parts), 0) + 1
        status[parts[0].strip()] = parts[1].strip() if len(parts) > 1 else ""

print("DONE_COLS " + json.dumps(cols))
sc = {}
for s in status.values():
    sc[s] = sc.get(s, 0) + 1
print("DONE_STATUS " + json.dumps(sc, ensure_ascii=False))

vi = {}
for path in sorted(glob.glob(os.path.join(QUEUE, "*.jsonl.zst"))):
    dctx = ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as rd:
            buf = rd.readall()
            for raw in buf.split(b"\n"):
                if not raw.strip():
                    continue
                rec = json.loads(raw)
                host = str(rec.get("host") or "").lower()
                if not is_vi(host):
                    continue
                e = vi.setdefault(host, {"queue": 0, "ok": 0})
                e["queue"] += 1
                if status.get(str(rec.get("id", "")).strip()) == "ok":
                    e["ok"] += 1

tot_q = sum(v["queue"] for v in vi.values())
tot_ok = sum(v["ok"] for v in vi.values())
print("VI queue=%d ok=%d chua_ok=%d" % (tot_q, tot_ok, tot_q - tot_ok))
out = {}
for h, v in sorted(vi.items(), key=lambda kv: -(kv[1]["queue"] - kv[1]["ok"])):
    if v["queue"] - v["ok"] > 0:
        out[h] = {"queue": v["queue"], "ok": v["ok"], "chua_ok": v["queue"] - v["ok"]}
print("VI_CHUA_OK " + json.dumps(out, ensure_ascii=False))
