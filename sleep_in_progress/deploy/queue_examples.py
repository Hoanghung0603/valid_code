import glob
import json
import os
import sys

from zstandard import ZstdDecompressor

BASE = r"D:\r2ai\data\crawl\async_recovery_20261009"
QUEUE = os.path.join(BASE, "queue")
HOST = (sys.argv[1] if len(sys.argv) > 1 else "qdnd.vn").lower()
LIMIT = int(sys.argv[2]) if len(sys.argv) > 2 else 15

found = 0
for path in sorted(glob.glob(os.path.join(QUEUE, "*.jsonl.zst"))):
    dctx = ZstdDecompressor()
    with open(path, "rb") as fh:
        with dctx.stream_reader(fh) as rd:
            buf = rd.readall()
            for raw in buf.split(b"\n"):
                if not raw.strip():
                    continue
                rec = json.loads(raw)
                if str(rec.get("host") or "").lower() != HOST:
                    continue
                print(json.dumps({
                    "id": rec.get("id"),
                    "url": rec.get("url"),
                    "group": rec.get("group"),
                    "reason": rec.get("reason"),
                    "attempts": rec.get("attempts"),
                    "route_hint": rec.get("route_hint"),
                    "shard": os.path.basename(path),
                }, ensure_ascii=False))
                found += 1
                if found >= LIMIT:
                    print("SHOWN %d" % found)
                    sys.exit(0)
print("SHOWN %d" % found)
