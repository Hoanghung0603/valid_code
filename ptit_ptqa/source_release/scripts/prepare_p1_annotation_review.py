"""Create a blinded P1 relevance-review pack from audited Kaggle pilot output.

The reviewer sees the official query and exact source passages, but not model
scores, model ranks, Exp122 rank, or the chunk-owner proxy.  A separate key
retains those fields for calibration after labels are frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from generate_translation_submission import sha256_file


ROOT = Path(__file__).resolve().parents[1]
PILOT_PAIRS = ROOT / "data/kaggle_inputs/p1_reranker_pilot_v1/p1_pairs.parquet"
PILOT_QUERIES = ROOT / "data/kaggle_inputs/p1_reranker_pilot_v1/p1_queries.parquet"
OUT = ROOT / "data/audits/p1_reranker_pilot_v1/annotation"
REVIEW_GUIDE = """# Hướng dẫn rà soát relevance P1

Hãy đánh giá mức độ các source passage hiển thị có thể hỗ trợ trả lời query. Chỉ dùng query và passages trong dòng đó; không tra cứu kiến thức ngoài, không suy luận rằng tài liệu đầy đủ hơn phần trích dẫn đang thấy.

- `direct`: passage có bằng chứng trực tiếp trả lời ít nhất một ý trọng tâm; không mâu thuẫn với bệnh, thuốc, đối tượng, tuổi, thai kỳ, phủ định hoặc ý định hỏi. Với query nhiều ý, chỉ cần hỗ trợ rõ một ý quan trọng.
- `related`: cùng chủ đề nhưng chỉ cung cấp thông tin nền, quá chung, thiếu bằng chứng để trả lời, hoặc lệch một ràng buộc quan trọng.
- `irrelevant`: sai chủ đề/ý định, không hỗ trợ query, chỉ chứa boilerplate, hoặc passage mâu thuẫn với điều query hỏi.

Nếu không đủ thông tin để quyết định, để `label` trống và ghi lý do ngắn trong `notes`; đừng đoán. Đừng sửa `annotation_id`, `query_id`, `doc_id` hay nội dung passages. Không có score/rank/split trong file này để giữ quá trình rà soát mù.

Đây là nhãn người rà soát trên một pool ứng viên được lấy mẫu, không phải qrels/gold labels của BTC và không đo được recall toàn corpus. Nhãn chỉ dùng để quyết định có đáng chạy một trial P1 hay không; điểm BTC vẫn là phép kiểm chứng cuối.
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True, help="Downloaded P1 Kaggle output ZIP")
    parser.add_argument("--pairs", type=Path, default=PILOT_PAIRS)
    parser.add_argument("--queries", type=Path, default=PILOT_QUERIES)
    parser.add_argument("--output-dir", type=Path, default=OUT)
    parser.add_argument("--seed", default="p1-review-v1")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def stable_hash(seed: str, value: object) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def allocate_stratified_holdout(queries: pd.DataFrame, size: int, seed: str) -> set[int]:
    """Allocate an exact-size, near-proportional holdout across all strata."""
    if not 0 < size < len(queries):
        raise ValueError("holdout size must be between zero and query count")
    counts = Counter(queries["stratum"].astype(str))
    target = {key: value * size / len(queries) for key, value in counts.items()}
    quotas = {key: int(value) for key, value in target.items()}
    remaining = size - sum(quotas.values())
    order = sorted(
        counts,
        key=lambda key: (-(target[key] - quotas[key]), stable_hash(seed, key)),
    )
    for key in order[:remaining]:
        quotas[key] += 1
    selected: set[int] = set()
    for key, group in queries.groupby(queries["stratum"].astype(str), sort=False):
        ranked = sorted(group.query_id.astype(int), key=lambda qid: stable_hash(seed, qid))
        selected.update(ranked[:quotas[str(key)]])
    if len(selected) != size:
        raise RuntimeError("Stratified holdout allocator did not produce the requested size")
    return selected


def load_doc_scores(path: Path) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        member = "p1_doc_scores.parquet"
        if archive.testzip() is not None or member not in archive.namelist():
            raise RuntimeError(f"Output ZIP is invalid or misses {member}")
        return pd.read_parquet(io.BytesIO(archive.read(member)))


def select_annotation_candidates(doc_scores: pd.DataFrame, queries: pd.DataFrame) -> pd.DataFrame:
    """Build a small rank-stratified queue for both models, without exposing ranks."""
    required = {
        "query_id", "doc_id", "qwen_rank", "bge_rank", "rank_disagreement", "owner_doc"
    }
    if not required.issubset(doc_scores.columns):
        raise RuntimeError(f"Document scores miss annotation fields: {sorted(required - set(doc_scores.columns))}")
    if doc_scores.duplicated(["query_id", "doc_id"]).any():
        raise RuntimeError("Document scores contain duplicate query-doc rows")
    reasons: dict[tuple[int, int], set[str]] = {}

    def add(row: pd.Series, reason: str) -> None:
        key = (int(row.query_id), int(row.doc_id))
        reasons.setdefault(key, set()).add(reason)

    for query_id, group in doc_scores.groupby("query_id", sort=True):
        for rank in (1, 40, 100, 180):
            selected = group[group.qwen_rank.astype(int) == rank]
            if len(selected) != 1:
                raise RuntimeError(f"Query {query_id} must have exactly one Qwen rank {rank}")
            add(selected.iloc[0], f"qwen_rank_{rank}")
        bge_top = group[group.bge_rank.astype(int) == 1]
        if len(bge_top) != 1:
            raise RuntimeError(f"Query {query_id} must have exactly one BGE rank 1")
        add(bge_top.iloc[0], "bge_rank_1")
        disagreement = group.sort_values(
            ["rank_disagreement", "doc_id"], ascending=[False, True], kind="stable"
        ).iloc[0]
        add(disagreement, "max_rank_disagreement")
        owners = group[group.owner_doc.astype(bool)]
        if not owners.empty:
            lowest_owner = owners.sort_values(
                ["qwen_rank", "doc_id"], ascending=[False, True], kind="stable"
            ).iloc[0]
            add(lowest_owner, "lowest_ranked_chunk_owner")

    rows = []
    by_key = doc_scores.set_index(["query_id", "doc_id"], drop=False)
    for (query_id, doc_id), sample_reasons in sorted(reasons.items()):
        record = by_key.loc[(query_id, doc_id)].to_dict()
        record["sampling_reasons"] = ";".join(sorted(sample_reasons))
        rows.append(record)
    queue = pd.DataFrame(rows).merge(
        queries[["query_id", "query"]], on="query_id", validate="many_to_one"
    )
    if queue.empty or queue.groupby("query_id").size().min() < 4:
        raise RuntimeError("Rank-stratified annotation queue is incomplete")
    return queue


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    paths = {
        "review_csv": output / "p1_annotation_blinded.csv",
        "review_parquet": output / "p1_annotation_blinded.parquet",
        "key": output / "p1_annotation_key.parquet",
        "instructions": output / "review_instructions.md",
        "manifest": output / "manifest.json",
    }
    if any(path.exists() for path in paths.values()) and not args.force:
        raise FileExistsError("Annotation outputs exist; use --force only before labels are entered")

    pairs = pd.read_parquet(args.pairs)
    queries = pd.read_parquet(args.queries)
    doc_scores = load_doc_scores(args.outputs)
    queue = select_annotation_candidates(doc_scores, queries)
    required_pairs = {"query_id", "doc_id", "passage_rank", "passage_text", "passage_sha256"}
    if not required_pairs.issubset(pairs.columns):
        raise RuntimeError("Pair table misses required review fields")
    if queue.duplicated(["query_id", "doc_id"]).any():
        raise RuntimeError("Annotation queue contains duplicate query-doc rows")

    passage_rows = pairs[pairs.set_index(["query_id", "doc_id"]).index.isin(
        queue.set_index(["query_id", "doc_id"]).index
    )].copy()
    passage_rows = passage_rows.sort_values(["query_id", "doc_id", "passage_rank"])
    passage_rows["passage_column"] = "passage_" + passage_rows.passage_rank.astype(str)
    texts = passage_rows.pivot(index=["query_id", "doc_id"], columns="passage_column", values="passage_text").reset_index()
    hashes = passage_rows.groupby(["query_id", "doc_id"]).passage_sha256.apply("|".join).rename("passage_hashes").reset_index()
    merged = queue.merge(texts, on=["query_id", "doc_id"], validate="one_to_one").merge(
        hashes, on=["query_id", "doc_id"], validate="one_to_one"
    )
    if len(merged) != len(queue):
        raise RuntimeError("Some annotation queue rows have no exact source passage")
    if "passage_2" not in merged:
        merged["passage_2"] = ""
    merged["passage_2"] = merged["passage_2"].fillna("")

    holdout_ids = allocate_stratified_holdout(queries, 40, args.seed)
    merged["split"] = merged.query_id.astype(int).map(lambda qid: "holdout" if qid in holdout_ids else "development")
    query_order = {
        qid: rank for rank, qid in enumerate(
            sorted(merged.query_id.astype(int).unique(), key=lambda qid: stable_hash(args.seed + ":query", qid))
        )
    }
    merged["_query_order"] = merged.query_id.map(query_order)
    merged["_doc_order"] = [stable_hash(args.seed + ":doc", f"{q}:{d}") for q, d in zip(merged.query_id, merged.doc_id, strict=True)]
    merged = merged.sort_values(["_query_order", "_doc_order"]).reset_index(drop=True)
    merged["annotation_id"] = [f"P1-{number:04d}" for number in range(1, len(merged) + 1)]

    review = merged[[
        "annotation_id", "query_id", "query", "doc_id", "passage_1", "passage_2"
    ]].copy()
    review["label"] = ""
    review["notes"] = ""
    forbidden = {"split", "owner_doc", "doc_rank", "bge_score", "qwen_score", "bge_rank", "qwen_rank", "rank_disagreement"}
    if forbidden.intersection(review.columns):
        raise RuntimeError("Blinded review accidentally exposes model/control features")

    key = merged.drop(columns=["_query_order", "_doc_order", "query", "passage_1", "passage_2"]).copy()
    output.mkdir(parents=True, exist_ok=True)
    review.to_csv(paths["review_csv"], index=False, encoding="utf-8-sig")
    review.to_parquet(paths["review_parquet"], index=False, compression="zstd")
    key.to_parquet(paths["key"], index=False, compression="zstd")
    paths["instructions"].write_text(REVIEW_GUIDE, encoding="utf-8")
    manifest = {
        "schema_version": "p1_annotation_review_v2",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_output_zip_sha256": sha256_file(args.outputs),
        "source_pairs_sha256": sha256_file(args.pairs),
        "source_queries_sha256": sha256_file(args.queries),
        "seed": args.seed,
        "rows": len(review),
        "queries": int(review.query_id.nunique()),
        "development_queries": int(merged.loc[merged.split == "development", "query_id"].nunique()),
        "holdout_queries": int(merged.loc[merged.split == "holdout", "query_id"].nunique()),
        "allowed_labels": ["direct", "related", "irrelevant"],
        "label_definitions": {
            "direct": "Passage directly supports at least one important query intent while respecting key constraints.",
            "related": "Same topic but lacks direct answer evidence or misses a key constraint.",
            "irrelevant": "Does not support the query, is off-intent, boilerplate-only, or contradicts it.",
        },
        "sampling_design": "per query: Qwen ranks 1/40/100/180, BGE rank 1, largest rank disagreement, and lowest-ranked chunk owner; deduplicated",
        "blinded_columns": review.columns.tolist(),
        "forbidden_features_absent": True,
        "artifacts": {
            name: {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for name, path in paths.items() if name != "manifest"
        },
    }
    paths["manifest"].write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
