"""Build one docs-only Exp144 candidate while preserving Exp143 chunks exactly."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from build_final_sprint_exp142_missing_parents import read_rows
    from build_p1_verified_docs_submission import package_lzma64
    from generate_translation_submission import sha256_file
except ModuleNotFoundError:
    from scripts.build_final_sprint_exp142_missing_parents import read_rows
    from scripts.build_p1_verified_docs_submission import package_lzma64
    from scripts.generate_translation_submission import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_exp143_p2a_long_chunks_add3.zip"
SPECS = ROOT / "data/query_specs/queryspec_production_v1/query_specs.parquet"
CORPUS = ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet"
VARIANTS = {
    "add5",
    "dense90",
    "dense130",
    "dynamic4",
    "dense130_owner",
    "fused130_owner",
    "fused130_owner_gated",
}


def append_owner_floor(selected: list[int], row: dict) -> list[int]:
    """Preserve every document that owns an Exp143 chunk, without duplicates."""
    seen = set(selected)
    result = list(selected)
    for chunk in row["relevant_chunks"]:
        doc_id = int(chunk["doc_id"])
        if doc_id not in seen:
            result.append(doc_id)
            seen.add(doc_id)
    return result


def fuse_current_and_dense(current: list[int], dense: list[int], top_k: int = 130) -> list[int]:
    """Weighted RRF: retain the proven Qwen ordering while admitting new dense hits."""
    scores: dict[int, float] = {}
    for rank, doc_id in enumerate(current, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (60 + rank)
    for rank, doc_id in enumerate(dense, 1):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.25 / (60 + rank)
    return sorted(scores, key=lambda doc_id: (-scores[doc_id], doc_id))[:top_k]


def load_rankings(path: Path) -> tuple[dict[int, list[int]], dict]:
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Exp144 output CRC failed")
        needed = {"exp144_rankings.npz", "exp144_manifest.json"}
        if not needed.issubset(archive.namelist()):
            raise RuntimeError(f"Missing ranking artifacts: {needed-set(archive.namelist())}")
        manifest = json.loads(archive.read("exp144_manifest.json"))
        with np.load(io.BytesIO(archive.read("exp144_rankings.npz")), allow_pickle=False) as arrays:
            query_ids = arrays["query_ids"].astype(np.int64)
            docs = arrays["docs"].astype(np.int64)
            scores = arrays["scores"].astype(np.float32)
    if query_ids.shape != (1200,) or docs.shape != (1200, 400) or scores.shape != docs.shape:
        raise RuntimeError(f"Unexpected ranking shapes: {query_ids.shape}, {docs.shape}, {scores.shape}")
    if len(set(map(int, query_ids))) != 1200 or not np.isfinite(scores).all():
        raise RuntimeError("Ranking IDs/scores invalid")
    return {int(qid): list(map(int, row)) for qid, row in zip(query_ids, docs, strict=True)}, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rankings-zip", type=Path, required=True)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"submission_exp144_{args.variant}_docs_exp143_chunks.zip"
    manifest_path = output.with_name(output.stem + "_manifest.json")
    stage = output.with_suffix(".json")
    building = output.with_name(output.stem + ".building.zip")
    if any(path.exists() for path in (output, manifest_path, stage, building)):
        raise FileExistsError("Refusing to overwrite Exp144 candidate/staging files")
    rankings, ranking_manifest = load_rankings(args.rankings_zip)
    official = set(pd.read_parquet(CORPUS, columns=["id"]).id.astype(int))
    specs = pd.read_parquet(SPECS, columns=["query_id", "status", "expected_max_docs"]).set_index("query_id")
    if len(specs) != 1200:
        raise RuntimeError("Expected 1,200 QuerySpecs")

    rows = docs_total = chunks_total = added = removed = changed_queries = 0
    target_sizes: list[int] = []
    try:
        with stage.open("x", encoding="utf-8", newline="\n") as dest:
            dest.write("[")
            for row in read_rows(CONTROL):
                qid = int(row["id"])
                current = list(map(int, row["relevant_docs"]))
                ranked = rankings[qid]
                if len(ranked) != len(set(ranked)) or not set(ranked).issubset(official):
                    raise RuntimeError(f"Invalid ranked docs for query {qid}")
                if args.variant == "add5":
                    current_set = set(current)
                    rescue = [doc for doc in ranked if doc not in current_set][:5]
                    selected = current + rescue
                elif args.variant == "dense90":
                    selected = ranked[:90]
                elif args.variant == "dense130":
                    selected = ranked[:130]
                elif args.variant == "dense130_owner":
                    selected = append_owner_floor(ranked[:130], row)
                elif args.variant == "fused130_owner":
                    selected = append_owner_floor(fuse_current_and_dense(current, ranked), row)
                elif args.variant == "fused130_owner_gated":
                    spec = specs.loc[qid]
                    selected = (append_owner_floor(fuse_current_and_dense(current, ranked), row)
                                if spec.status == "ok" else current)
                else:
                    spec = specs.loc[qid]
                    expected = int(spec.expected_max_docs) if spec.status == "ok" and pd.notna(spec.expected_max_docs) else 30
                    target = min(180, max(40, 4 * expected))
                    selected = ranked[:target]
                if not selected or len(selected) != len(set(selected)):
                    raise RuntimeError(f"Empty/duplicate selection for query {qid}")
                old = set(current); new = set(selected)
                added += len(new - old); removed += len(old - new)
                changed_queries += int(selected != current)
                row["relevant_docs"] = selected
                target_sizes.append(len(selected))
                if rows:
                    dest.write(",\n")
                dest.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                rows += 1; docs_total += len(selected); chunks_total += len(row["relevant_chunks"])
            dest.write("\n]\n")
        if rows != 1200 or chunks_total != 127_271:
            raise RuntimeError(f"Coverage/chunk preservation failed: {rows=} {chunks_total=}")
        staged_zip = package_lzma64(stage, output)
        if staged_zip.stat().st_size >= 100_000_000:
            raise RuntimeError(f"Candidate exceeds decimal 100 MB: {staged_zip.stat().st_size}")
        with zipfile.ZipFile(staged_zip) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("Candidate ZIP layout/CRC failed")
        os.replace(staged_zip, output)
    except Exception:
        output.unlink(missing_ok=True); building.unlink(missing_ok=True)
        raise
    finally:
        stage.unlink(missing_ok=True)
    manifest = {
        "experiment": "Exp144",
        "variant": args.variant,
        "treatment": "docs-only QuerySpec/BM25 union reranked with original-VI exact BGE-M3",
        "control": "Exp143 chunks preserved exactly",
        "owner_floor": args.variant in {"dense130_owner", "fused130_owner", "fused130_owner_gated"},
        "queries_changed_vs_control": changed_queries,
        "queries": rows,
        "docs": docs_total,
        "chunks": chunks_total,
        "docs_added_vs_control": added,
        "docs_removed_vs_control": removed,
        "docs_per_query": pd.Series(target_sizes).describe().to_dict(),
        "rankings_zip_sha256": sha256_file(args.rankings_zip),
        "ranking_manifest": ranking_manifest,
        "control_sha256": sha256_file(CONTROL),
        "submission_zip_sha256": sha256_file(output),
        "submission_zip_bytes": output.stat().st_size,
        "btc_score": None,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
