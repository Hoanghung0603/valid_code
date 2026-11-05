"""Print the first N urls from named queue shards. Read-only."""
import json
import sys
from pathlib import Path

import zstandard as zstd

HOSTS = (
    "vov_vn",
    "zysjonline_com",
    "zydcd_com",
    "familydoctor_com_cn",
    "nhathuoclongchau_com_vn",
    "article_iiyi_com",
    "bingli_iiyi_com",
    "pmc_ecm_healthblog_beta_pharmacity_io",
)
N = 2


def first_urls(path: Path, n: int):
    dctx = zstd.ZstdDecompressor()
    out = []
    with path.open("rb") as fh:
        with dctx.stream_reader(fh) as reader:
            buf = b""
            while len(out) < n:
                chunk = reader.read(1 << 16)
                if not chunk:
                    break
                buf += chunk
                *lines, buf = buf.split(b"\n")
                for raw in lines:
                    if not raw.strip():
                        continue
                    row = json.loads(raw.decode("utf-8", "ignore"))
                    out.append({"id": row["id"], "url": row["url"], "host": row["host"]})
                    if len(out) >= n:
                        break
    return out


def main() -> int:
    qdir = Path(sys.argv[1])
    rows = []
    for stem in HOSTS:
        rows.extend(first_urls(qdir / f"host-{stem}.jsonl.zst", N))
    small = qdir / "smalls-00.jsonl.zst"
    tamanh = []
    if small.exists():
        dctx = zstd.ZstdDecompressor()
        with small.open("rb") as fh:
            with dctx.stream_reader(fh) as reader:
                buf = b""
                while len(tamanh) < N:
                    chunk = reader.read(1 << 20)
                    if not chunk:
                        break
                    buf += chunk
                    *lines, buf = buf.split(b"\n")
                    for raw in lines:
                        if not raw.strip():
                            continue
                        row = json.loads(raw.decode("utf-8", "ignore"))
                        if row.get("host") == "tamanhhospital.vn":
                            tamanh.append({"id": row["id"], "url": row["url"],
                                           "host": row["host"]})
                        if len(tamanh) >= N:
                            break
    rows.extend(tamanh)
    for r in rows:
        print(json.dumps(r, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
