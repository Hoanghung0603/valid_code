"""Build a docs-only P1 trial from full-corpus reranker scores.

The builder never changes Exp122 chunks.  It retains the top reranked control
documents plus every control document that owns at least one Exp122 chunk.
Choosing ``--keep-top`` is deliberately explicit: it must come from P1
pilot/annotation calibration. Human labels cover a sampled candidate pool,
not BTC qrels, and cannot establish corpus-wide recall or score improvement.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import numpy as np

try:  # Support direct CLI execution and package-style test imports.
    from generate_translation_submission import iter_json_array, sha256_file
except ModuleNotFoundError:
    from scripts.generate_translation_submission import iter_json_array, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_exp122_exp121_add19_miss.zip"
CONTROL_SHA256 = "8c7f221cfafa401dfb5435df7da06464999c4b60594e69bc7726bb4ffb379886"
SCORE_MEMBER = "p1_doc_scores.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True, help="Full p1_doc_scores parquet or output ZIP")
    parser.add_argument("--calibration", type=Path, required=True, help="calibration.json from calibrate_p1_reranker.py")
    parser.add_argument("--model", choices=("bge", "qwen4b"), required=True)
    parser.add_argument("--keep-top", type=int, required=True)
    parser.add_argument("--control", type=Path, default=CONTROL)
    parser.add_argument("--out-prefix", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--allow-control-hash-mismatch", action="store_true")
    parser.add_argument(
        "--allow-unreviewed-exploratory",
        action="store_true",
        help="Explicitly build the weak/mixed exploratory policy without the human-review gate",
    )
    parser.add_argument(
        "--allow-exploratory-policy-override", action="store_true",
        help="Explicitly test a different model/cutoff; never mark it calibrated or human-reviewed",
    )
    parser.add_argument("--exploratory-rationale", help="Required when overriding the calibration policy")
    return parser.parse_args()


def load_control(path: Path) -> list[dict]:
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("Control ZIP must contain one valid root submission.json")
        with archive.open("submission.json") as binary, io.TextIOWrapper(binary, encoding="utf-8") as text:
            rows = list(iter_json_array(text))
    if len(rows) != 1200 or len({int(row["id"]) for row in rows}) != 1200:
        raise RuntimeError("Control must contain 1,200 unique query IDs")
    return rows


def load_scores(path: Path) -> tuple[pd.DataFrame, str]:
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            if archive.testzip() is not None or SCORE_MEMBER not in archive.namelist():
                raise RuntimeError(f"Score ZIP is invalid or misses {SCORE_MEMBER}")
            payload = archive.read(SCORE_MEMBER)
        scores = pd.read_parquet(io.BytesIO(payload))
        payload_hash = hashlib.sha256(payload).hexdigest()
    else:
        scores = pd.read_parquet(path)
        payload_hash = sha256_file(path)

    # Full-run outputs use the explicit model key (qwen4b_score), while pilot
    # and calibration tables use the shorter qwen_score alias. Normalize only
    # the schema name; preserve the original score values and provenance hash.
    if "qwen_score" not in scores.columns and "qwen4b_score" in scores.columns:
        scores = scores.rename(columns={"qwen4b_score": "qwen_score"})
    return scores, payload_hash


def package_lzma64(json_path: Path, zip_path: Path, force: bool = False) -> Path:
    """Stage a one-member ZIP with the same 64 MiB LZMA dictionary as Exp122."""
    seven_zip = shutil.which("7z")
    if not seven_zip:
        raise RuntimeError("7z is required to package the 64 MiB LZMA submission")
    temp_zip = zip_path.with_name(zip_path.stem + ".building.zip")
    if temp_zip.exists():
        if not force:
            raise FileExistsError(temp_zip)
        temp_zip.unlink()
    with json_path.open("rb") as source:
        subprocess.run(
            [seven_zip, "a", "-tzip", "-mm=LZMA", "-md=64m", "-mx=9",
             str(temp_zip.resolve()), "-sisubmission.json"],
            stdin=source, capture_output=True, text=True, check=True,
        )
    return temp_zip


def retained_docs(
    control_docs: list[int],
    chunk_owner_docs: set[int],
    ranked_docs: list[int],
    keep_top: int,
) -> list[int]:
    """Return a stable control-order subset with reranked and owner protections."""
    if not 1 <= keep_top < len(control_docs):
        raise ValueError("keep_top must be between 1 and len(control_docs)-1")
    control_set = set(control_docs)
    if set(ranked_docs) != control_set or len(ranked_docs) != len(control_docs):
        raise ValueError("Reranker coverage differs from the control document set")
    keep = set(ranked_docs[:keep_top]) | (chunk_owner_docs & control_set)
    return [doc_id for doc_id in control_docs if doc_id in keep]


def resolve_calibration_policy(calibration: dict, allow_unreviewed: bool) -> tuple[str, str, int]:
    human_reviewed = bool(calibration.get("human_review_gate", {}).get("eligible_for_pool_trial"))
    if human_reviewed:
        model = calibration.get("human_only_recommended_model")
        keep_top = calibration.get("human_only_recommended_keep_top")
        mode = "human_reviewed_candidate_pool_trial"
    elif allow_unreviewed:
        model = calibration.get("exploratory_recommended_model")
        keep_top = calibration.get("exploratory_recommended_keep_top")
        mode = "unreviewed_exploratory"
    else:
        raise RuntimeError(
            "Calibration has not passed the operational human-review gate; "
            "do not build a trial unless --allow-unreviewed-exploratory is intentional"
        )
    if model not in {"bge", "qwen4b"} or not isinstance(keep_top, int) or not 1 <= keep_top <= 215:
        raise RuntimeError(f"Calibration policy is invalid or a no-op: model={model}, keep_top={keep_top}")
    return mode, model, keep_top


def select_requested_policy(
    calibration: dict, model: str, keep_top: int, allow_unreviewed: bool,
    allow_override: bool, rationale: str | None,
) -> str:
    if allow_override:
        if not allow_unreviewed or not rationale or not rationale.strip():
            raise RuntimeError(
                "Exploratory override requires --allow-unreviewed-exploratory and a nonempty rationale"
            )
        if calibration.get("human_review_gate", {}).get("eligible_for_pool_trial"):
            raise RuntimeError("Do not override a human-reviewed policy using the unreviewed path")
        return "uncalibrated_exploratory"
    if rationale:
        raise RuntimeError("--exploratory-rationale requires --allow-exploratory-policy-override")
    policy_mode, calibrated_model, calibrated_keep_top = resolve_calibration_policy(
        calibration, allow_unreviewed
    )
    if model != calibrated_model or keep_top != calibrated_keep_top:
        raise RuntimeError(
            f"Requested policy {model}/top{keep_top} differs from "
            f"calibration {calibrated_model}/top{calibrated_keep_top} ({policy_mode})"
        )
    return policy_mode


def main() -> None:
    args = parse_args()
    if not 1 <= args.keep_top < 216:
        raise ValueError("--keep-top must be in [1, 215]; 216 would be a no-op")
    control_hash = sha256_file(args.control)
    if control_hash != CONTROL_SHA256 and not args.allow_control_hash_mismatch:
        raise RuntimeError(f"Unexpected Exp122 control hash: {control_hash}")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    policy_mode = select_requested_policy(
        calibration, args.model, args.keep_top, args.allow_unreviewed_exploratory,
        args.allow_exploratory_policy_override, args.exploratory_rationale,
    )

    json_path = args.out_prefix.with_suffix(".json")
    zip_path = args.out_prefix.with_suffix(".zip")
    manifest_path = args.out_prefix.with_name(args.out_prefix.name + "_manifest.json")
    if any(path.exists() for path in (json_path, zip_path, manifest_path)) and not args.force:
        raise FileExistsError("Output exists; pass --force only after reviewing provenance")

    control = load_control(args.control)
    scores, score_payload_hash = load_scores(args.scores)
    score_col = "bge_score" if args.model == "bge" else "qwen_score"
    required = {"query_id", "doc_id", score_col}
    if not required.issubset(scores.columns):
        raise RuntimeError(f"Scores miss columns: {sorted(required - set(scores.columns))}")
    if scores[list(required)].isna().any().any():
        raise RuntimeError("Scores contain null identifiers or model scores")
    if not np.isfinite(scores[score_col].to_numpy(dtype=float)).all():
        raise RuntimeError("Scores contain non-finite model scores")
    if scores.duplicated(["query_id", "doc_id"]).any():
        raise RuntimeError("Scores contain duplicate query-doc rows")

    by_query = {int(qid): group for qid, group in scores.groupby("query_id", sort=False)}
    output: list[dict] = []
    docs_before = docs_after = owner_docs = changed_queries = 0
    for row in control:
        query_id = int(row["id"])
        control_docs = [int(doc_id) for doc_id in row["relevant_docs"]]
        if len(control_docs) != 216 or len(set(control_docs)) != 216:
            raise RuntimeError(f"Control query {query_id} does not have 216 unique docs")
        if query_id not in by_query or len(by_query[query_id]) != 216:
            raise RuntimeError(f"Scores do not have 216 docs for query {query_id}")
        ranked = (
            by_query[query_id]
            .sort_values([score_col, "doc_id"], ascending=[False, True])["doc_id"]
            .astype(int)
            .tolist()
        )
        owners = {int(chunk["doc_id"]) for chunk in row["relevant_chunks"]}
        docs = retained_docs(control_docs, owners, ranked, args.keep_top)
        if docs != control_docs:
            changed_queries += 1
        docs_before += len(control_docs)
        docs_after += len(docs)
        owner_docs += len(owners & set(control_docs))
        output.append({"id": query_id, "relevant_docs": docs, "relevant_chunks": row["relevant_chunks"]})

    if set(by_query) != {int(row["id"]) for row in control}:
        raise RuntimeError("Score query IDs differ from control")
    if changed_queries == 0:
        raise RuntimeError("P1 treatment is a no-op")

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(output, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    # Exp122 uses ZIP/LZMA with a 64 MiB dictionary. Python zipfile's default
    # 8 MiB dictionary makes this verbatim-chunk payload exceed the 100 MB cap.
    temp_zip = package_lzma64(json_path, zip_path, args.force)
    # Stream the packaged payload.  Full submissions decompress to hundreds of
    # megabytes, so archive.read()+json.loads() can exhaust RAM even when the
    # ZIP itself is valid and below the upload limit.
    with zipfile.ZipFile(temp_zip) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("Generated submission ZIP failed layout/CRC validation")
        if archive.infolist()[0].compress_type != zipfile.ZIP_LZMA:
            raise RuntimeError("Generated submission ZIP did not use LZMA")
        with archive.open("submission.json") as binary, io.TextIOWrapper(binary, encoding="utf-8") as text:
            generated_rows = iter_json_array(text)
            validated_rows = 0
            for base, treatment in zip(control, generated_rows, strict=True):
                if base["id"] != treatment["id"] or base["relevant_chunks"] != treatment["relevant_chunks"]:
                    raise RuntimeError("Query IDs or chunks changed relative to Exp122")
                if not set(treatment["relevant_docs"]).issubset(set(base["relevant_docs"])):
                    raise RuntimeError("Treatment added documents outside Exp122")
                internal_owners = {
                    int(chunk["doc_id"]) for chunk in base["relevant_chunks"]
                } & set(base["relevant_docs"])
                if not internal_owners.issubset(set(treatment["relevant_docs"])):
                    raise RuntimeError("Treatment removed a protected chunk-owning document")
                validated_rows += 1
            if validated_rows != 1200:
                raise RuntimeError(f"Generated submission has {validated_rows} rows, expected 1,200")
    if temp_zip.stat().st_size >= 100_000_000:
        raise RuntimeError("Generated submission exceeds the 100 MB upload limit")
    os.replace(temp_zip, zip_path)

    is_human_reviewed = policy_mode == "human_reviewed_candidate_pool_trial"
    manifest = {
        "schema_version": (
            "p1_human_reviewed_docs_trial_v1" if is_human_reviewed else
            "p1_uncalibrated_docs_trial_v1" if policy_mode == "uncalibrated_exploratory" else
            "p1_model_label_exploratory_docs_trial_v1"
        ),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "control_zip_sha256": control_hash,
        "score_source": str(args.scores.resolve()),
        "score_payload_sha256": score_payload_hash,
        "calibration_sha256": sha256_file(args.calibration),
        "model_labels_sha256": calibration.get("model_labels_sha256"),
        "human_labels_sha256": calibration.get("human_labels_sha256"),
        "calibration_policy_mode": policy_mode,
        "calibration_exploratory_recommendation": {
            "model": calibration.get("exploratory_recommended_model"),
            "keep_top": calibration.get("exploratory_recommended_keep_top"),
        },
        "exploratory_policy_override_rationale": args.exploratory_rationale,
        "human_review_gate_passed": is_human_reviewed,
        "promotion_status": "candidate_for_btc_evaluation" if is_human_reviewed else "exploratory_only_not_promoted",
        "calibration_operational_human_review_gate": calibration.get("human_review_gate"),
        "evaluation_scope": calibration.get("evaluation_scope", "unspecified"),
        "btc_qrels_available": calibration.get("btc_qrels_available"),
        "synthetic_dev_set_used": calibration.get("synthetic_dev_set_used"),
        "model": args.model,
        "score_column": score_col,
        "keep_top": args.keep_top,
        "changed_queries": changed_queries,
        "docs_before": docs_before,
        "docs_after": docs_after,
        "protected_internal_owner_query_docs": owner_docs,
        "chunks_exactly_equal_control": True,
        "submission_json_sha256": sha256_file(json_path),
        "submission_zip_sha256": sha256_file(zip_path),
        "submission_zip_bytes": zip_path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
