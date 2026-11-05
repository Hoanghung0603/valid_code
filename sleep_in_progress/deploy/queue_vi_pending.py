import glob
import json
import os
import sys

from zstandard import ZstdDecompressor

BASE = r"D:\r2ai\data\crawl\async_recovery_20261009"
QUEUE = os.path.join(BASE, "queue")
DONE = os.path.join(BASE, "state", "done_ids.tsv")


def is_vi(host):
    h = (host or "").lower()
    return h.endswith(".vn") or h == "vinmec.com"


def main():
    done = set()
    with open(DONE, "r", encoding="utf-8", errors="replace") as f:
        first = f.readline()
        f.seek(0)
        parts = first.rstrip("\n").split("\t")
        print("DONE_HEAD " + json.dumps(parts[:5], ensure_ascii=False))
        done.add(parts[0].strip())
        for line in f:
            if not line.strip():
                continue
            done.add(line.split("\t", 1)[0].strip())
    print("DONE_IDS %d" % len(done))

    per = {}
    tot_vi = 0
    tot_vi_done = 0
    tot_all = 0
    tot_all_done = 0
    sample_keys = None
    for path in sorted(glob.glob(os.path.join(QUEUE, "*.jsonl.zst"))):
        dctx = ZstdDecompressor()
        with open(path, "rb") as fh:
            with dctx.stream_reader(fh) as rd:
                buf = rd.readall()
                for raw in buf.split(b"\n"):
                    if not raw.strip():
                        continue
                    rec = json.loads(raw)
                    if sample_keys is None:
                        sample_keys = sorted(rec.keys())
                    rid = str(rec.get("id", "")).strip()
                    host = str(rec.get("host") or "").lower()
                    tot_all += 1
                    hit = rid in done
                    if hit:
                        tot_all_done += 1
                    if not is_vi(host):
                        continue
                    tot_vi += 1
                    e = per.setdefault(host, {"queue": 0, "done": 0, "pending": 0})
                    e["queue"] += 1
                    if hit:
                        e["done"] += 1
                        tot_vi_done += 1
                    else:
                        e["pending"] += 1

    top = sorted(per.items(), key=lambda kv: -kv[1]["pending"])[:20]
    print("KEYS " + json.dumps(sample_keys))
    print("ALL queue=%d done=%d pending=%d" % (tot_all, tot_all_done, tot_all - tot_all_done))
    print("VI queue=%d done=%d pending=%d" % (tot_vi, tot_vi_done, tot_vi - tot_vi_done))
    print("VI_HOSTS " + json.dumps(
        {h: v for h, v in sorted(per.items(), key=lambda kv: -kv[1]["queue"])},
        ensure_ascii=False,
    ))
    print("VI_TOP_PENDING " + json.dumps(
        [{"host": h, **v} for h, v in top], ensure_ascii=False
    ))


if __name__ == "__main__":
    main()
