"""`medret` command line. Pipeline stages get their own commands as they are implemented."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from medret.config import load_config

app = typer.Typer(no_args_is_help=True, add_completion=False)

ConfigOpt = Annotated[
    Path | None, typer.Option("--config", "-c", help="YAML config (default: configs/default.yaml)")
]


@app.command()
def download(config: ConfigOpt = None) -> None:
    """Download the ViBioMIR dataset snapshot from HuggingFace into data/raw."""
    from medret.data import download_dataset

    typer.echo(f"Downloaded to {download_dataset(load_config(config))}")


@app.command()
def manifest(
    workers: Annotated[int | None, typer.Option(help="Processes for hashing shards (default: all CPUs)")] = None,
    config: ConfigOpt = None,
) -> None:
    """Index crawl_dir/pages shards into crawl_dir/manifest.parquet and aliases.parquet (URL twins)."""
    from medret.crawl import build_aliases, build_manifest
    from medret.crawl.manifest import ALIASES_FILE, MANIFEST_FILE
    from medret.data import load_corpus

    cfg = load_config(config)
    crawl_dir = cfg.paths.crawl_dir
    m = build_manifest(crawl_dir, workers)
    m.to_parquet(crawl_dir / MANIFEST_FILE, index=False)
    corpus = load_corpus(cfg)
    aliases = build_aliases(corpus, m)
    aliases.to_parquet(crawl_dir / ALIASES_FILE, index=False)

    ok = m["error"].eq("")
    typer.echo(f"fetched ids: {len(m):,}  ok: {ok.sum():,}  by source: {m[ok]['source'].value_counts().to_dict()}")
    typer.echo(f"errors: {m.loc[~ok, 'error'].str.split(':').str[0].value_counts().to_dict()}")
    covered = aliases["fetched_id"].notna()
    typer.echo(
        f"corpus ids with OK text (incl. URL twins): {covered.sum():,} / {len(corpus):,} ({covered.mean():.1%}); "
        f"canonical keys: {aliases.loc[covered, 'key'].nunique():,} / {aliases['key'].nunique():,}"
    )


WorkersOpt = Annotated[int, typer.Option(help="Worker processes")]


@app.command()
def templates(
    per_host: Annotated[int, typer.Option(help="Pages sampled per host")] = 800,
    workers: WorkersOpt = 8,
    config: ConfigOpt = None,
) -> None:
    """Learn per-host boilerplate templates on a sample of crawled pages (run before `parse`)."""
    from medret.parse.run import TEMPLATES_FILE, learn_host_templates

    cfg = load_config(config)
    out = cfg.paths.parsed_dir / TEMPLATES_FILE
    t = learn_host_templates(cfg.paths.crawl_dir, out, per_host=per_host, workers=workers)
    typer.echo(f"{len(t)} hosts, {sum(len(x.lines) for x in t.values()):,} template lines -> {out}")


@app.command()
def parse(
    workers: WorkersOpt = 8,
    limit_shards: Annotated[int | None, typer.Option(help="Parse only this many shards (smoke test)")] = None,
    config: ConfigOpt = None,
) -> None:
    """HTML shards -> data/parsed/documents (resumable: finished shards are skipped)."""
    from medret.parse.run import parse_corpus

    cfg = load_config(config)
    parse_corpus(cfg.paths.crawl_dir, cfg.paths.parsed_dir, workers=workers, limit_shards=limit_shards,
                 log=typer.echo)


@app.command()
def chunk(workers: WorkersOpt = 8, config: ConfigOpt = None) -> None:
    """data/parsed/documents -> data/chunks (resumable)."""
    from medret.chunk.run import chunk_corpus
    from medret.parse.run import DOCS_SUBDIR

    cfg = load_config(config)
    chunk_corpus(cfg.paths.parsed_dir / DOCS_SUBDIR, cfg.paths.chunks_dir, tokenizer=cfg.embed.model,
                 workers=workers, log=typer.echo)


@app.command("export-embed")
def export_embed(
    rows_per_shard: Annotated[int, typer.Option(help="Unique inputs per output shard (~9 min per T4)")] = 50_000,
    config: ConfigOpt = None,
) -> None:
    """Write deduplicated BGE-M3 inputs (one per embed_sha1) to data/embed_inputs for the GPU runs."""
    from medret.index.export import export_embed_inputs
    from medret.parse.run import DOCS_SUBDIR

    cfg = load_config(config)
    export_embed_inputs(cfg.paths.parsed_dir / DOCS_SUBDIR, cfg.paths.chunks_dir, cfg.paths.data_dir / "embed_inputs",
                        rows_per_shard=rows_per_shard, log=typer.echo)


@app.command("build-meta")
def build_meta_cmd(config: ConfigOpt = None) -> None:
    """DuckDB metadata store (documents, chunks + input_id, URL aliases) -> <index_dir>/meta.duckdb."""
    from medret.crawl.manifest import ALIASES_FILE
    from medret.index.meta import DB_FILE, build_meta
    from medret.parse.run import DOCS_SUBDIR

    cfg = load_config(config)
    build_meta(cfg.paths.index_dir / DB_FILE, cfg.paths.parsed_dir / DOCS_SUBDIR, cfg.paths.chunks_dir,
               cfg.paths.data_dir / "embed_inputs", cfg.paths.crawl_dir / ALIASES_FILE,
               memory_limit=cfg.index.meta_memory_limit, log=typer.echo)


@app.command("build-scope")
def build_scope_cmd(config: ConfigOpt = None) -> None:
    """Docs / embedding inputs of the current corpus file (dataset.corpus_file) -> <index_dir>/scope_*.npy."""
    from medret.data import load_corpus
    from medret.index.scope import build_scope

    cfg = load_config(config)
    build_scope(cfg.paths.index_dir, load_corpus(cfg), log=typer.echo)


@app.command("build-bm25")
def build_bm25_cmd(
    lang: Annotated[list[str], typer.Option(help="vi, zh and/or en")] = ["vi", "zh", "en"],  # noqa: B006
    scoped: Annotated[bool, typer.Option(help="Only docs in the corpus scope (build-scope)")] = True,
    config: ConfigOpt = None,
) -> None:
    """Tantivy BM25 index per language over all chunks -> <index_dir>/bm25/<lang>."""
    from medret.index.lexical import build_bm25
    from medret.index.scope import SCOPE_DOCS
    from medret.parse.run import DOCS_SUBDIR

    cfg = load_config(config)
    scope = cfg.paths.index_dir / SCOPE_DOCS if scoped else None
    if scope is not None and not scope.exists():
        raise typer.BadParameter(f"{scope} missing: run `medret build-scope` (or pass --no-scoped)")
    for lg in lang:
        build_bm25(cfg.paths.parsed_dir / DOCS_SUBDIR, cfg.paths.chunks_dir, cfg.paths.index_dir / "bm25" / lg, lg,
                   workers=cfg.index.bm25_workers, scope_file=scope, log=typer.echo)


@app.command("build-dense")
def build_dense_cmd(
    skip_faiss: Annotated[bool, typer.Option(help="Only copy new shards into the memmap")] = False,
    scoped: Annotated[bool, typer.Option(help="Index only vectors of the corpus scope (build-scope)")] = True,
    config: ConfigOpt = None,
) -> None:
    """Copy finished embedding shards into the fp16 memmap, then (re)build the FAISS IVF-PQ index."""
    import pyarrow.parquet as pq

    from medret.index.dense import assemble_vectors, build_ivfpq
    from medret.index.export import SHARD_GLOB

    cfg = load_config(config)
    run = cfg.index.dense_run
    # count shard rows only: the folder also holds the Kaggle dataset-metadata.json
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in (cfg.paths.data_dir / "embed_inputs").glob(SHARD_GLOB))
    out = cfg.paths.index_dir / "dense" / run
    emb = cfg.paths.data_dir / "embeddings"
    # the main run plus incremental runs (crawl increments embedded by a separate kernel): <run>-inc*/
    for src in [emb / run, *sorted(emb.glob(f"{run}-inc*"))]:
        if src.is_dir():
            assemble_vectors(src, out, n, log=typer.echo)
    if not skip_faiss:
        from medret.index.scope import load_scope

        scope = load_scope(cfg.paths.index_dir) if scoped else None
        if scoped and scope is None:
            raise typer.BadParameter("scope missing: run `medret build-scope` (or pass --no-scoped)")
        build_ivfpq(out, factory=cfg.index.faiss_factory, run=run, scope_inputs=scope[1] if scope else None,
                    log=typer.echo)


@app.command("search-bm25")
def search_bm25_cmd(
    query: str,
    lang: Annotated[str, typer.Option(help="vi or zh index")] = "vi",
    k: int = 5,
    config: ConfigOpt = None,
) -> None:
    """Debug: top-k chunks of one BM25 index for a query, with their verbatim text."""
    from medret.index.lexical import BM25Index
    from medret.index.meta import DB_FILE, MetaStore

    cfg = load_config(config)
    hits = BM25Index(cfg.paths.index_dir / "bm25" / lang).search(query, k)
    meta = MetaStore(cfg.paths.index_dir / DB_FILE).chunks([c for c, _ in hits]).set_index("chunk_id")
    for cid, score in hits:
        r = meta.loc[cid]
        typer.echo(f"\n[{score:.2f}] chunk {cid} doc {r.doc_id} · {r.title[:80]}\n{r.chunk_text[:300]}")


@app.command("encode-queries")
def encode_queries_cmd(
    input: Annotated[Path | None, typer.Option(help="Parquet with an `id` column (default: query.parquet)")] = None,
    text_col: Annotated[str, typer.Option(help="Column holding the query text (e.g. query_zh for translations)")] = "query",
    out: Annotated[Path | None, typer.Option(help="Output .npz (ids, vecs); default data/queries/qvec_<col>.npz")] = None,
    threads: Annotated[int | None, typer.Option(help="CPU threads for torch")] = None,
    config: ConfigOpt = None,
) -> None:
    """BGE-M3 query vectors on CPU (no instruction, CLS + L2), cached in data/cache/qvec.sqlite."""
    import time

    import numpy as np
    import pandas as pd

    from medret.data import load_queries
    from medret.retrieve.query_encoder import QueryEncoder

    cfg = load_config(config)
    df = pd.read_parquet(input) if input else load_queries(cfg)
    out = out or cfg.paths.data_dir / "queries" / f"qvec_{text_col}.npz"
    enc = QueryEncoder(cfg.embed.model, cache_path=cfg.paths.data_dir / "cache" / "qvec.sqlite", threads=threads)
    t0 = time.time()
    vecs = enc.encode(df[text_col].fillna("").astype(str).tolist())
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=df["id"].to_numpy(np.int64), vecs=vecs)
    typer.echo(f"{len(df):,} queries -> {vecs.shape} in {time.time() - t0:.1f}s -> {out}")


@app.command("translate-queries")
def translate_queries_cmd(
    model: Annotated[str | None, typer.Option(help="Chat model id (default: llm.translate_model)")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N queries (smoke test)")] = None,
    config: ConfigOpt = None,
) -> None:
    """Vietnamese queries -> Simplified Chinese -> data/queries/query_zh__<model>.parquet (cached)."""
    import time

    import pyarrow as pa
    import pyarrow.parquet as pq

    from medret.data import load_queries
    from medret.retrieve.translate import DEV_ONLY, PROMPT_VERSION, Translator

    cfg = load_config(config)
    model = model or cfg.llm.translate_model
    df = load_queries(cfg)
    df = df.head(limit) if limit else df
    tr = Translator(model, cache_path=cfg.paths.data_dir / "cache" / "translate.sqlite", threads=cfg.llm.threads)
    t0 = time.time()
    df = df.assign(query_zh=tr.translate(df["query"].tolist()))
    out = cfg.paths.data_dir / "queries" / f"query_zh__{model}.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df[["id", "query", "query_zh"]], preserve_index=False)
    meta = {b"model": model.encode(), b"prompt_version": PROMPT_VERSION.encode(),
            b"dev_only": str(model in DEV_ONLY).encode()}
    pq.write_table(table.replace_schema_metadata({**(table.schema.metadata or {}), **meta}), out)
    flag = "  [DEV ONLY: not allowed in submissions]" if model in DEV_ONLY else ""
    typer.echo(f"{len(df):,} queries translated by {model} in {time.time() - t0:.0f}s, {tr.tokens:,} tokens -> {out}{flag}")


@app.command("run")
def run_cmd(
    name: Annotated[str, typer.Option(help="Run name -> data/runs/<name>/")],
    queries: Annotated[Path | None, typer.Option(help="Parquet with id, query (default: competition queries)")] = None,
    limit: Annotated[int | None, typer.Option(help="Only the first N queries (smoke test)")] = None,
    allow_dev_models: Annotated[bool, typer.Option(help="Allow > 15B dev-only models (test runs only)")] = False,
    tau_doc: float = 0.1, k_max: int = 20, tau_chunk: float = 0.3, top_m: int = 100,
    n_candidates: int = 300, k_per_channel: int = 200,
    zh_tau: float | None = None, max_zh_docs: int | None = None,
    channels: Annotated[str, typer.Option(help="Comma list of retrieval channels")] = "bm25_vi,bm25_zh,dense_vi,dense_zh",
    config: ConfigOpt = None,
) -> None:
    """Full inference: translate -> encode -> retrieve -> rerank -> select -> submission.json + zip."""
    import pandas as pd

    from medret.data import load_queries
    from medret.pipeline import RerankConfig, RunConfig, run_pipeline
    from medret.retrieve.pipeline import RetrieveConfig
    from medret.select.selector import SelectConfig

    cfg = load_config(config)
    q = pd.read_parquet(queries) if queries else load_queries(cfg)
    q = q.head(limit) if limit else q
    rc = RunConfig(retrieve=RetrieveConfig(channels=channels.split(","), n_candidates=n_candidates,
                                           k_per_channel=k_per_channel),
                   rerank=RerankConfig(top_m=top_m),
                   select=SelectConfig(tau_doc=tau_doc, k_max=k_max, tau_chunk=tau_chunk, zh_tau=zh_tau,
                                       max_zh_docs=max_zh_docs),
                   translate_model=cfg.llm.translate_model, allow_dev_models=allow_dev_models)
    run_pipeline(cfg, name, q, rc, log=typer.echo)


@app.command("select")
def select_cmd(
    name: Annotated[str, typer.Option(help="Existing run in data/runs/<name>/ (needs reranked.parquet)")],
    out_name: Annotated[str | None, typer.Option(help="Write to another run dir (default: overwrite)")] = None,
    tau_doc: float = 0.1, k_min: int = 1, k_max: int = 20, tau_chunk: float = 0.3, max_chunks_per_doc: int = 3,
    include_twins: bool = False, zh_tau: float | None = None, max_zh_docs: int | None = None, full_doc_top: int = 0,
    rrf_weight: float = 0.1, chunk_docs: int | None = None, full_doc_max_chars: int | None = None,
    config: ConfigOpt = None,
) -> None:
    """Re-run only the selection step of a finished run with other thresholds (no model calls)."""
    import pandas as pd

    from medret.index.meta import DB_FILE, MetaStore
    from medret.pipeline import select_and_write
    from medret.select.selector import SelectConfig
    from medret.submit import pack_submission, validate_submission

    cfg = load_config(config)
    src = cfg.paths.runs_dir / name
    out = cfg.paths.runs_dir / (out_name or name)
    out.mkdir(parents=True, exist_ok=True)
    qids = pd.read_parquet(src / "queries.parquet")["id"].tolist()
    sc = SelectConfig(tau_doc=tau_doc, k_min=k_min, k_max=k_max, tau_chunk=tau_chunk,
                      max_chunks_per_doc=max_chunks_per_doc, include_twins=include_twins, zh_tau=zh_tau,
                      max_zh_docs=max_zh_docs, full_doc_top=full_doc_top, rrf_weight=rrf_weight,
                      chunk_docs=chunk_docs, full_doc_max_chars=full_doc_max_chars)
    path = select_and_write(cfg, out, qids, pd.read_parquet(src / "reranked.parquet"), sc,
                            MetaStore(cfg.paths.index_dir / DB_FILE), log=typer.echo)
    errors = validate_submission(path, qids)
    typer.echo("validate: OK" if not errors else f"validate: {errors[:5]}")
    if not errors:  # zip named after the run directory
        typer.echo(f"zip: {pack_submission(path, out / f'{out.name}.zip')}")


@app.command("make-dev")
def make_dev_cmd(
    n: Annotated[int, typer.Option(help="Number of synthetic queries")] = 300,
    name: Annotated[str, typer.Option(help="data/dev/<name>.json + <name>_queries.parquet")] = "syn_v1",
    seed: int = 0,
    config: ConfigOpt = None,
) -> None:
    """Synthetic dev set: Vietnamese questions generated from sampled chunks (|G| = 1; measures retrieval)."""
    from medret.eval.devset import make_devset
    from medret.index.meta import DB_FILE, MetaStore

    cfg = load_config(config)
    stats = make_devset(MetaStore(cfg.paths.index_dir / DB_FILE), cfg.paths.dev_dir, name, n, cfg.llm.translate_model,
                        cfg.paths.data_dir / "cache" / "translate.sqlite", seed=seed, threads=cfg.llm.threads)
    typer.echo(f"{stats} -> {cfg.paths.dev_dir / name}.json")


@app.command("dev-report")
def dev_report_cmd(
    name: Annotated[str, typer.Option(help="Run in data/runs/<name>/ made on the dev queries")],
    dev: Annotated[str, typer.Option(help="Dev set name in data/dev/")] = "syn_v1",
    config: ConfigOpt = None,
) -> None:
    """Retrieval diagnostics (candidate recall per channel, rank after rerank) + official F2 on a dev set."""
    import pandas as pd

    from medret.eval import evaluate, get_tokenizer
    from medret.eval.retrieval_report import retrieval_report
    from medret.submit import load_results

    cfg = load_config(config)
    run_dir = cfg.paths.runs_dir / name
    rep = retrieval_report(run_dir, pd.read_parquet(cfg.paths.dev_dir / f"{dev}_queries.parquet"))
    f2 = evaluate(load_results(run_dir / "submission.json"), load_results(cfg.paths.dev_dir / f"{dev}.json"),
                  get_tokenizer(cfg.eval.tokenizer), relevance_threshold=cfg.eval.relevance_threshold,
                  near_duplicate_threshold=cfg.eval.near_duplicate_threshold, beta=cfg.eval.beta).summary()
    rep["official_f2_on_dev"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in f2.items()}
    (run_dir / f"dev_report_{dev}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False), encoding="utf-8")
    typer.echo(json.dumps(rep, indent=1, ensure_ascii=False))


@app.command()
def validate(
    submission: Path,
    check_queries: Annotated[bool, typer.Option(help="Require every id in query.parquet")] = True,
    check_corpus: Annotated[bool, typer.Option(help="Require doc ids to exist in links_corpus")] = False,
    config: ConfigOpt = None,
) -> None:
    """Check a submission JSON against the official format."""
    from medret.data import load_corpus, load_queries
    from medret.submit import validate_submission

    cfg = load_config(config)
    query_ids = load_queries(cfg)["id"].tolist() if check_queries else None
    corpus_ids = set(load_corpus(cfg)["id"].astype(str)) if check_corpus else None
    errors = validate_submission(submission, query_ids, corpus_ids)
    for e in errors[:50]:
        typer.echo(f"- {e}")
    if errors:
        typer.echo(f"{len(errors)} problem(s)")
        raise typer.Exit(1)
    typer.echo("OK")


@app.command()
def pack(submission: Path, out: Annotated[Path | None, typer.Option()] = None) -> None:
    """Zip a submission JSON (single file at the archive root) for the dashboard."""
    from medret.submit import pack_submission

    typer.echo(pack_submission(submission, out))


@app.command()
def evaluate(
    submission: Path,
    qrels: Path,
    tokenizer: Annotated[str | None, typer.Option(help="HF tokenizer id, or 'whitespace'")] = None,
    report: Annotated[Path | None, typer.Option(help="Write per-query scores as JSON")] = None,
    config: ConfigOpt = None,
) -> None:
    """Score a submission against a qrels file (same JSON shape as a submission)."""
    from dataclasses import asdict

    from medret.eval import evaluate as run_eval
    from medret.eval import get_tokenizer
    from medret.submit import load_results

    cfg = load_config(config).eval
    result = run_eval(
        load_results(submission),
        load_results(qrels),
        get_tokenizer(tokenizer or cfg.tokenizer),
        relevance_threshold=cfg.relevance_threshold,
        near_duplicate_threshold=cfg.near_duplicate_threshold,
        beta=cfg.beta,
    )
    for k, v in result.summary().items():
        typer.echo(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(asdict(result), ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    app()
