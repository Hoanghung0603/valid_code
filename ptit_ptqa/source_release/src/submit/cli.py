"""Command-line interface for the Multilingual Retrieval & Submission Pipeline."""

import argparse
import logging
import os
import sys

from src.retrieval.query_encoder import BGEM3QueryEncoder
from src.retrieval.index_repository import IndexRepository
from src.retrieval.searcher import VectorSearcher
from src.reranking.cross_encoder import CrossEncoderReranker
from src.pipeline.submission_exporter import SubmissionExporter
from src.pipeline.submission_pipeline import SubmissionPipeline

# Ensure UTF-8 output on Windows terminals
if sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

DEFAULT_QUERY_FILE = "data/raw/btc_provided/query_translated.parquet"
FALLBACK_QUERY_FILE = "data/raw/btc_provided/query.parquet"


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Multilingual Retrieval with BGE-M3 (Dense, Sparse, Hybrid, RRF).")
    parser.add_argument("--query_file", type=str, default=None,
                        help="Path to query parquet file (default: query_translated.parquet if exists else query.parquet)")
    parser.add_argument("--mode", type=str, choices=["dense", "sparse", "hybrid"], default="hybrid",
                        help="Retrieval mode: dense, sparse, or hybrid (default: hybrid)")
    parser.add_argument("--fusion", type=str, choices=["linear", "rrf"], default="rrf",
                        help="Fusion strategy in hybrid mode: linear (score sum) or rrf (reciprocal rank fusion, default: rrf)")
    parser.add_argument("--dense_weight", type=float, default=1.0,
                        help="Weight for dense retrieval (default: 1.0)")
    parser.add_argument("--sparse_weight", type=float, default=1.0,
                        help="Weight for sparse lexical retrieval (default: 1.0)")
    parser.add_argument("--multi_sparse_weights", type=float, nargs="+", default=None,
                        help="List of sparse weights to test simultaneously in one pass, e.g. --multi_sparse_weights 0.5 1.0 1.5")
    parser.add_argument("--rrf_k", type=int, default=60,
                        help="Constant k for Reciprocal Rank Fusion: 1 / (k + rank) (default: 60)")
    parser.add_argument("--zh_weight", type=float, default=1.0,
                        help="Scaling factor for translated Chinese lexical tokens (default: 1.0)")
    parser.add_argument("--en_weight", type=float, default=0.5,
                        help="Scaling factor for translated English lexical tokens (default: 0.5)")
    parser.add_argument("--cand_pool", type=int, default=150,
                        help="Number of candidates to retain per shard per modality (default: 150)")
    parser.add_argument("--cand_buffer", type=int, default=400,
                        help="Intermediate candidate buffer size per query across shards (default: 400)")
    parser.add_argument("--top_docs", type=int, default=100,
                        help="Maximum relevant documents to output per query (default: 100)")
    parser.add_argument("--top_chunks", type=int, default=98,
                        help="Maximum relevant chunks to output per query (default: 98 for optimal zip <100MB)")
    parser.add_argument("--score_threshold", type=float, default=None,
                        help="Similarity threshold for filtering candidates (default: None)")
    parser.add_argument("--multi_thresholds", type=float, nargs="+", default=None,
                        help="List of score thresholds to evaluate simultaneously, e.g. --multi_thresholds 0.65 0.70 0.73")
    parser.add_argument("--threshold_metric", type=str, choices=["dense", "hybrid"], default="dense",
                        help="Metric to apply threshold against: dense (cosine) or hybrid (default: dense)")
    parser.add_argument("--min_docs", type=int, default=5,
                        help="Minimum number of documents to retain per query even if score < threshold (default: 5)")
    parser.add_argument("--min_chunks", type=int, default=5,
                        help="Minimum number of chunks to retain per query even if score < threshold (default: 5)")
    parser.add_argument("--rerank", action="store_true",
                        help="Enable 2-stage Cross-Encoder reranking")
    parser.add_argument("--rerank_cand_docs", type=int, default=200,
                        help="Number of candidate documents to retrieve in stage 1 for reranking (default: 200)")
    parser.add_argument("--reranker_model", type=str, default="BAAI/bge-reranker-v2-m3",
                        help="Cross-Encoder model for reranking")
    parser.add_argument("--reranker_batch_size", type=int, default=64,
                        help="Batch size for cross-encoder reranking")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON filename")
    parser.add_argument("--btc_links_file", type=str, default="data/raw/btc_provided/links_corpus_vi_en_zh.parquet",
                        help="Path to BTC links parquet file for filtering out removed documents (default: data/raw/btc_provided/links_corpus_vi_en_zh.parquet)")
    parser.add_argument("--no_filter_btc", action="store_true",
                        help="Disable filtering candidates by BTC valid links")
    parser.add_argument("--index_base", type=str,
                        default="data/indices/new_dataset" if os.path.exists("data/indices/new_dataset/index/chunks") else "data/indices",
                        help="Base directory containing search index shards (default: data/indices/new_dataset if available)")
    parser.add_argument("--no_faiss", action="store_true",
                        help="Disable FAISS and force fallback to standard shard-by-shard search")
    parser.add_argument("--no_multilingual", action="store_true",
                        help="Disable multilingual translation auto-merge and search strictly in native query text")
    return parser.parse_args(args)


def main():
    args = parse_args()
    mode = args.mode
    fusion = args.fusion if mode == "hybrid" else "linear"

    # Select query file
    query_path = args.query_file
    if query_path is None:
        if os.path.exists(DEFAULT_QUERY_FILE):
            query_path = DEFAULT_QUERY_FILE
        else:
            query_path = FALLBACK_QUERY_FILE

    # Configure run targets
    if args.multi_sparse_weights:
        mode = "hybrid"
        configs = []
        for sw in args.multi_sparse_weights:
            sw_tag = f"{fusion}_w{str(sw).replace('.', '')}"
            configs.append({
                "dense_w": args.dense_weight,
                "sparse_w": sw,
                "tag": sw_tag,
                "out_json": f"submission_bgem3_{sw_tag}.json",
                "out_zip": f"submission_bgem3_{sw_tag}.zip",
            })
    elif args.multi_thresholds:
        mode = "hybrid"
        configs = []
        configs.append({
            "dense_w": args.dense_weight,
            "sparse_w": args.sparse_weight,
            "threshold": None,
            "tag": "top100_unconstrained",
            "out_json": "submission_bgem3_top100.json",
            "out_zip": "submission_bgem3_top100.zip",
        })
        for th in args.multi_thresholds:
            th_tag = f"thresh_{str(th).replace('.', '')}"
            configs.append({
                "dense_w": args.dense_weight,
                "sparse_w": args.sparse_weight,
                "threshold": th,
                "tag": th_tag,
                "out_json": f"submission_bgem3_hybrid_w02_{th_tag}.json",
                "out_zip": f"submission_bgem3_hybrid_w02_{th_tag}.zip",
            })
    else:
        if args.output:
            out_json = args.output
        elif args.rerank:
            out_json = f"submission_bgem3_rerank_cand{args.rerank_cand_docs}_top{args.top_docs}.json"
        elif args.score_threshold is not None:
            th_tag = f"thresh_{str(args.score_threshold).replace('.', '')}"
            out_json = f"submission_bgem3_{mode}_{fusion}_{th_tag}.json"
        else:
            out_json = f"submission_bgem3_{mode}_{fusion}.json"
        out_zip = out_json.replace(".json", ".zip")
        configs = [{
            "dense_w": args.dense_weight,
            "sparse_w": args.sparse_weight,
            "threshold": args.score_threshold,
            "tag": "single",
            "out_json": out_json,
            "out_zip": out_zip,
        }]

    print("=" * 70)
    print(f"🚀 MULTILINGUAL BGE-M3 RETRIEVAL PIPELINE (OOP / SOLID)")
    print(f"   Mode: {mode.upper()} | Fusion: {fusion.upper()}")
    print(f"   Query file: {query_path}")
    print(f"   Index base: {args.index_base}")
    print(f"   Rerank Enabled: {args.rerank}")
    print(f"   Configs to evaluate: {len(configs)}")
    print("=" * 70, flush=True)

    # 1. Dependency Injection setup
    index_repo = IndexRepository(index_base=args.index_base)
    valid_doc_ids = None if args.no_filter_btc else index_repo.load_valid_doc_ids(args.btc_links_file)

    query_encoder = BGEM3QueryEncoder()

    faiss_cand_dir = os.path.join(args.index_base, "index") if os.path.exists(os.path.join(args.index_base, "index", "faiss_index_flat_ip.bin")) else args.index_base
    faiss_bin_path = os.path.join(faiss_cand_dir, "faiss_index_flat_ip.bin")

    if os.path.exists(faiss_bin_path) and not args.no_faiss:
        logger.info(f"⚡ FAISS index detected at {faiss_bin_path}! Activating FAISSSearcher (Memory-Mapped)...")
        from src.retrieval.faiss_searcher import FAISSSearcher
        searcher = FAISSSearcher(
            index_dir=faiss_cand_dir,
            cand_pool=args.cand_pool,
            cand_buffer=args.cand_buffer
        )
    else:
        logger.info("Using standard VectorSearcher (shard-by-shard scan)...")
        searcher = VectorSearcher(
            index_repository=index_repo,
            cand_pool=args.cand_pool,
            cand_buffer=args.cand_buffer
        )
    exporter = SubmissionExporter()

    reranker = None
    if args.rerank:
        reranker = CrossEncoderReranker(
            model_name=args.reranker_model,
            cand_docs_limit=args.rerank_cand_docs
        )

    pipeline = SubmissionPipeline(
        encoder=query_encoder,
        searcher=searcher,
        index_repository=index_repo,
        exporter=exporter,
        reranker=reranker
    )

    # 2. Load Queries
    queries = pipeline.load_queries(query_path, enable_multilingual=not args.no_multilingual)

    # 3. Execute Pipeline
    pipeline.execute(
        queries=queries,
        configs=configs,
        mode=mode,
        fusion_type=fusion,
        zh_weight=args.zh_weight,
        en_weight=args.en_weight,
        rrf_k=args.rrf_k,
        top_docs=args.top_docs,
        top_chunks=args.top_chunks,
        score_threshold=args.score_threshold,
        threshold_metric=args.threshold_metric,
        min_docs=args.min_docs,
        min_chunks=args.min_chunks,
        valid_doc_ids=valid_doc_ids,
        rerank_batch_size=args.reranker_batch_size
    )

    print("\n" + "=" * 70)
    print("✨ TẤT CẢ CÁC BỘ SUBMISSION ĐÃ SẴN SÀNG!")
    for cfg in configs:
        sz = os.path.getsize(cfg["out_zip"]) / (1024 * 1024) if os.path.exists(cfg["out_zip"]) else 0
        print(f" • Config '{cfg['tag']}': {cfg['out_zip']} ({sz:.2f} MB)")
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
