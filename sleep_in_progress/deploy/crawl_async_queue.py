"""Build per-host sharded queue for the async broad recovery job (LOCAL build step).

Inputs (read-only): local links_corpus.parquet (sha dee1153f...), downloaded
trung state logs (read-only copies), ask20 jsonl.
Output: queue shards <out>/queue/<group>-<host>.jsonl.zst with rows
  {id, url, host, group, reason, attempts, route_hint}
plus queue_manifest.json {per-shard sha256, counts, totals}.

Exact scope rule (user permission):
  queue_ids = corpus_ids(trung scope: host NOT cnkang/120ask)
              - trung_state terminal `ok` (exact literal)
              - ask20 (20 already harvested)
  permanent INCLUDED (short-content re-extract required).
  cnkang/120ask excluded entirely (original b660 scope, outside this job).

Groups / route hints (reason-aware, baked per row):
  - never_gaveup   : never attempted, host gave_up before -> proxy_first
  - tier4_never    : tier4 hard hosts, never attempted -> proxy_first
  - unres_blocked  : last blocked -> proxy_first, long cooldown
  - unres_redirect : qdnd TooManyRedirects -> direct_first (redirect inspect)
  - unres_transient: transient-only -> direct_first
  - permanent      : terminal permanent -> direct_first + txt-fallback flag

Sharding: one shard per big host (>=50k), mids grouped 1 file each host,
smalls merged into smalls-*.zst (<=50k rows each). Deterministic order by id.
"""
import argparse
import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse

import zstandard as zstd

B660_HOSTS = {"cnkang.com", "120ask.com"}
HARD_HOSTS = {"zysjonline.com", "nhathuoclongchau.com.vn", "tamanhhospital.vn",
              "vov.vn", "pmc-ecm-healthblog.beta.pharmacity.io",
              "article.iiyi.com", "bingli.iiyi.com"}
BIG_THRESHOLD = 50000
SMALL_MERGE_CAP = 50000
SCHEMA = ("id", "url", "host", "group", "reason", "attempts", "route_hint")


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def read_state_last(paths):
    """last outcome per id; exact literals kept verbatim."""
    last = {}
    for p in paths:
        with open(p, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.rstrip("\n").rstrip("\r")
                if not line or "\t" not in line:
                    continue
                sid, _, outcome = line.partition("\t")
                try:
                    i = int(sid)
                except ValueError:
                    continue
                if outcome.strip():
                    last[i] = outcome.strip()
    return last


def classify_row(sid, outcome):
    """-> (group, reason, attempts-info). outcome None = never attempted."""
    if outcome is None:
        return None  # decided by tier below
    if outcome == "permanent":
        return ("permanent", "permanent_short_or_410", 1)
    if outcome.startswith("blocked"):
        return ("unres_blocked", "last_blocked", 1)
    if "TooManyRedirects" in outcome:
        return ("unres_redirect", "last_redirect_loop", 1)
    return ("unres_transient", "last_transient", 1)


def route_hint_for(group):
    if group in ("never_gaveup", "tier4_never", "unres_blocked"):
        return "proxy_first"
    return "direct_first"


def iter_corpus(corpus):
    """Stream id,url from parquet in record batches (low RAM: never build a
    full DataFrame; 4.4M-row corpus must not require a 1GB arrow block)."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(corpus)
    seen = 0
    prev_id = None
    for batch in pf.iter_batches(batch_size=65536, columns=["id", "url"]):
        ids = batch.column("id").to_pylist()
        urls = batch.column("url").to_pylist()
        for sid, url in zip(ids, urls):
            if sid is None or url is None:
                raise ValueError("corpus IDs/urls must be non-null")
            sid = int(sid)
            if prev_id is not None and sid == prev_id:
                raise ValueError(f"duplicate corpus id {sid}")
            prev_id = sid
            yield sid, str(url)
            seen += 1
    if not seen:
        raise ValueError("empty corpus")


def build(corpus, state_paths, ask20_path, out_dir):
    """Disk-bucketing build: O(1) row memory in phase 1, one host file at a
    time in phase 2 (peak RAM ~= largest host, not the 1.36M-row queue)."""
    out_dir = Path(out_dir)
    if out_dir.exists():
        raise FileExistsError(f"Refuse existing output: {out_dir}")
    qdir = out_dir / "queue"
    tmpdir = out_dir / "tmp_buckets"
    qdir.mkdir(parents=True)
    tmpdir.mkdir(parents=True)
    ask20 = set()
    if ask20_path is not None:
        for line in Path(ask20_path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                ask20.add(json.loads(line)["id"])
    last = read_state_last(state_paths)
    counts = Counter()
    queued_ids: set = set()
    corpus_rows = 0
    b660_rows = 0
    ok_excluded = 0
    bucket_files = {}
    # ---- phase 1: stream corpus -> per-host plain bucket files (disk, not RAM)
    for sid, url in iter_corpus(corpus):
        corpus_rows += 1
        host = host_of(url)
        if host in B660_HOSTS:
            b660_rows += 1
            continue
        outcome = last.get(sid)
        if outcome == "ok" or sid in ask20:
            ok_excluded += 1
            continue  # exact dedup: original-ok + ask20 + harvested recovery
        if sid in queued_ids:
            raise ValueError(f"queue id collision {sid}")
        queued_ids.add(sid)
        if outcome is None:
            group = "tier4_never" if host in HARD_HOSTS else "never_gaveup"
            reason = "tier4_never_attempted" if group == "tier4_never" else "gaveup_host_never_attempted"
            attempts = 0
        else:
            g = classify_row(sid, outcome)
            group, reason, attempts = g
        row = {"id": sid, "url": str(url), "host": host, "group": group,
               "reason": reason, "attempts": attempts,
               "route_hint": route_hint_for(group)}
        counts[group] += 1
        safe = host.replace(".", "_").replace("-", "_")
        fh = bucket_files.get(safe)
        if fh is None:
            fh = (tmpdir / f"bucket_{safe}.jsonl").open("a", encoding="utf-8")
            bucket_files[safe] = fh
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    for fh in bucket_files.values():
        fh.close()
    # ---- phase 2: one host bucket at a time -> zstd queue shards
    manifest = {"shards": [], "groups": dict(counts),
                "total": sum(counts.values()), "hosts": len(bucket_files),
                "ask20_excluded": len(ask20), "corpus_rows": corpus_rows,
                "b660_scope_rows": b660_rows,
                "ok_ask20_excluded": ok_excluded,
                "queue_ids_unique": True}
    comp = zstd.ZstdCompressor(level=9)
    small_buf, small_idx, small_n = [], 0, 0

    def write_shard(name, rows):
        path = qdir / name
        with path.open("xb") as fh:
            with comp.stream_writer(fh) as zw:
                for r in rows:
                    zw.write((json.dumps(r, ensure_ascii=False) + "\n").encode("utf-8"))
                zw.flush(zstd.FLUSH_FRAME)
        fh2 = hashlib.sha256()
        with path.open("rb") as fh:
            while True:
                b = fh.read(1 << 20)
                if not b:
                    break
                fh2.update(b)
        manifest["shards"].append({"file": name, "host": rows[0]["host"] if len({r['host'] for r in rows}) == 1 else "mixed",
                                   "rows": len(rows), "sha256": fh2.hexdigest(),
                                   "groups": dict(Counter(r["group"] for r in rows))})

    for safe in sorted(bucket_files):
        bpath = tmpdir / f"bucket_{safe}.jsonl"
        rows = []
        with bpath.open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
        rows.sort(key=lambda r: r["id"])
        bpath.unlink()  # free disk as we go
        if len(rows) >= 5000:
            write_shard(f"host-{safe}.jsonl.zst", rows)
        else:
            small_buf.extend(rows)
            small_n += len(rows)
            if small_n >= SMALL_MERGE_CAP:
                small_buf.sort(key=lambda r: r["id"])
                write_shard(f"smalls-{small_idx:02d}.jsonl.zst", small_buf)
                small_idx += 1
                small_buf, small_n = [], 0
    if small_buf:
        small_buf.sort(key=lambda r: r["id"])
        write_shard(f"smalls-{small_idx:02d}.jsonl.zst", small_buf)
    try:
        tmpdir.rmdir()
    except OSError:
        pass
    if manifest["total"] != len(queued_ids):
        raise ValueError("queue total != unique queued ids")
    (out_dir / "queue_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"total": manifest["total"], "hosts": manifest["hosts"],
                      "shards": len(manifest["shards"]), "groups": manifest["groups"],
                      "corpus_rows": corpus_rows}, ensure_ascii=False))
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--state", nargs="+", required=True)
    ap.add_argument("--ask20", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    build(a.corpus, a.state, a.ask20, a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
