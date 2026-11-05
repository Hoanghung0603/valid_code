"""Stage 1 - fetch every URL in links_corpus.parquet and store raw responses.

Planned: async fetcher with per-domain rate limits, resumable state keyed by doc id, compressed
shards under data/crawl, bulk/API sources for large domains (PubMed, PMC, Wikipedia), and an
archive fallback (Wayback / Common Crawl) for dead links. See docs/design/architecture.md section 2.1.

Implemented: `manifest` indexes page shards produced by the team crawler (crawl_dir/pages).
"""

from medret.crawl.manifest import build_aliases, build_manifest, iter_pages

__all__ = ["build_aliases", "build_manifest", "iter_pages"]
