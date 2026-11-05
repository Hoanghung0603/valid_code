"""Corpus-scale parse: learn host templates on a sample, then turn every OK page into a document record.

Outputs (paths under cfg.paths.parsed_dir):
  templates.json                      host -> {n_docs, lines, headings}, learned before parsing
  documents/<site>/<part>.parquet     one output per crawl page shard, so runs resume shard by shard

Document columns follow docs/design/architecture.md section 3, plus host and text_sha1 (exact-duplicate key).
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from medret.crawl.manifest import MANIFEST_FILE
from medret.data.urls import host_language, host_of
from medret.parse.clean import CleanConfig, HostTemplates, clean_blocks, learn_templates, normalize_block
from medret.parse.document import assemble
from medret.parse.html import decode_html, extract

EXTRACTOR_VERSION = "resiliparse-full+main-flag/clean-v8"
TEMPLATES_FILE = "templates.json"
DOCS_SUBDIR = "documents"
BATCH_ROWS = 256  # html rows decoded at a time, bounds worker memory on 500 KB news pages

DOC_SCHEMA = pa.schema([
    ("doc_id", pa.int64()),
    ("url", pa.string()),
    ("host", pa.string()),
    ("lang", pa.string()),
    ("title", pa.string()),
    ("text", pa.string()),
    ("blocks", pa.list_(pa.struct([("start", pa.int32()), ("end", pa.int32()), ("kind", pa.string()),
                                   ("heading_path", pa.string())]))),
    ("n_chars", pa.int32()),
    ("text_sha1", pa.string()),
    ("dup_group", pa.int64()),
    ("extractor_version", pa.string()),
])


def ok_manifest(crawl_dir: Path) -> pd.DataFrame:
    m = pd.read_parquet(crawl_dir / MANIFEST_FILE, columns=["doc_id", "url", "content_type", "shard", "offset",
                                                            "length", "error"])
    m = m[m["error"].eq("") & m["length"].gt(0)].drop(columns=["error", "length"])
    m["host"] = m["url"].map(host_of)
    return m.reset_index(drop=True)


def _iter_rows(crawl_dir: Path, shard: str, rows: pd.DataFrame):
    """(row, raw html) for the given manifest rows of one shard, reading the html column in batches."""
    want = dict(zip(rows["offset"].astype(int), rows.itertuples(index=False), strict=True))
    pf = pq.ParquetFile(crawl_dir / shard)
    base = 0
    for batch in pf.iter_batches(columns=["html"], batch_size=BATCH_ROWS):
        col = batch.column(0)
        for i in range(len(col)):
            r = want.get(base + i)
            if r is not None:
                yield r, col[i].as_py()
        base += len(col)


def _extract_shard(crawl_dir: Path, shard: str, rows: pd.DataFrame) -> list[tuple[str, list]]:
    out = []
    for r, raw in _iter_rows(crawl_dir, shard, rows):
        try:
            out.append((r.host, extract(decode_html(raw, r.content_type)[0]).blocks))
        except Exception:  # noqa: BLE001 - a broken page must not stop template learning
            continue
    return out


def learn_host_templates(crawl_dir: Path, out_file: Path, per_host: int = 800, workers: int = 8,
                         seed: int = 0) -> dict[str, HostTemplates]:
    """Sample up to `per_host` OK pages per host, extract them and learn line / heading templates."""
    m = ok_manifest(crawl_dir)
    sample = pd.concat(g.sample(min(len(g), per_host), random_state=seed) for _, g in m.groupby("host"))
    blocks_by_host: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_extract_shard, crawl_dir, s, g) for s, g in sample.groupby("shard")]
        for f in as_completed(futs):
            for host, blocks in f.result():
                blocks_by_host.setdefault(host, []).append(blocks)
    templates = {h: learn_templates(b) for h, b in blocks_by_host.items()}
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps({h: {"n_docs": t.n_docs, "lines": sorted(t.lines), "headings": sorted(t.headings)}
                                    for h, t in sorted(templates.items())}, ensure_ascii=False, indent=0),
                        encoding="utf-8")
    return templates


def load_templates(path: Path) -> dict[str, HostTemplates]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {h: HostTemplates(v["n_docs"], set(v["lines"]), set(v["headings"])) for h, v in raw.items()}


def parse_page(raw: bytes, content_type: str, templates: HostTemplates | None, cfg: CleanConfig) -> dict:
    ex = extract(decode_html(raw, content_type)[0])
    blocks, _ = clean_blocks(ex.blocks, ex.h1, templates, cfg)
    text, spans = assemble(blocks)
    title = next((b.text for b in blocks if b.kind == "h1"), "") or normalize_block(ex.title)
    return {"title": title.replace("\n", " "), "text": text, "blocks": spans}


def _parse_shard(crawl_dir: str, shard: str, rows: pd.DataFrame, out_path: str, templates_file: str) -> tuple[int, int]:
    crawl = Path(crawl_dir)
    templates = load_templates(Path(templates_file))
    cfg = CleanConfig()
    recs, failed = [], 0
    for r, raw in _iter_rows(crawl, shard, rows):
        try:
            d = parse_page(raw, r.content_type, templates.get(r.host), cfg)
        except Exception:  # noqa: BLE001 - one malformed page is recorded as empty, not fatal
            failed += 1
            d = {"title": "", "text": "", "blocks": []}
        recs.append({
            "doc_id": int(r.doc_id), "url": r.url, "host": r.host, "lang": host_language(r.host), **d,
            "n_chars": len(d["text"]), "text_sha1": hashlib.sha1(d["text"].encode("utf-8")).hexdigest(),
            "dup_group": None, "extractor_version": EXTRACTOR_VERSION,
        })
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(recs, schema=DOC_SCHEMA), tmp, compression="zstd")
    tmp.replace(out)
    return len(recs), failed


def parse_corpus(crawl_dir: Path, parsed_dir: Path, workers: int = 8, limit_shards: int | None = None,
                 log=print) -> None:
    """Parse every OK page; shards whose output already exists are skipped (resumable)."""
    templates_file = parsed_dir / TEMPLATES_FILE
    if not templates_file.exists():
        raise FileNotFoundError(f"{templates_file} missing: run `medret templates` first")
    m = ok_manifest(crawl_dir)
    jobs = []
    for shard, g in m.groupby("shard"):
        out = parsed_dir / DOCS_SUBDIR / Path(shard).relative_to("pages")
        if not out.exists():
            jobs.append((shard, g, out))
    jobs = jobs[:limit_shards] if limit_shards else jobs
    log(f"{len(jobs)} shards to parse ({sum(len(g) for _, g, _ in jobs):,} pages)")
    t0, done, pages, failed = time.time(), 0, 0, 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_parse_shard, str(crawl_dir), s, g, str(o), str(templates_file)) for s, g, o in jobs]
        for f in as_completed(futs):
            n, bad = f.result()
            done, pages, failed = done + 1, pages + n, failed + bad
            if done % 100 == 0 or done == len(jobs):
                rate = pages / (time.time() - t0)
                log(f"{done}/{len(jobs)} shards, {pages:,} pages, {failed} failed, {rate:,.0f} pages/s")
