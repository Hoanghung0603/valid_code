"""Append a small set of verbatim full-source parents for missing chunk owners."""
from __future__ import annotations

import hashlib
import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd

try:
    from build_p1_verified_docs_submission import package_lzma64
    from generate_translation_submission import iter_json_array, sha256_file
except ModuleNotFoundError:
    from scripts.build_p1_verified_docs_submission import package_lzma64
    from scripts.generate_translation_submission import iter_json_array, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "submission_qwen4b_chunks_keep90_full.zip"
SOURCE = ROOT / "submission_best_axes_docs_joint_chunks_top130.zip"
QWEN_DOCS = ROOT / "data/audits/p1_full_qwen4b_merged/p1_full_qwen4b_merged.parquet"
BGE_DOCS = ROOT / "data/audits/p1_full_bge_merged/p1_full_bge_merged.parquet"
INDEX = ROOT / "data/indices/new_dataset/index"
CORPUS = ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet"
MAX_PER_QUERY = 5
RRF_K = 60
QUERY_COUNT = 1200


def read_rows(path: Path):
    with zipfile.ZipFile(path) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError(f"Invalid ZIP source: {path}")
        with archive.open("submission.json") as raw, io.TextIOWrapper(raw, encoding="utf-8") as text:
            yield from iter_json_array(text)


def sha256_stream(stream) -> str:
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def load_rank_maps(path: Path, rank_col: str) -> dict[int, dict[int, int]]:
    frame = pd.read_parquet(path, columns=["query_id", "doc_id", rank_col])
    if (len(frame) != 259_200 or frame.groupby("query_id").size().ne(216).any()
            or frame.duplicated(["query_id", "doc_id"]).any() or frame[rank_col].isna().any()):
        raise RuntimeError(f"Invalid full ranking cache: {path}")
    return {
        int(qid): dict(zip(g.doc_id.astype(int), g[rank_col].astype(int), strict=True))
        for qid, g in frame.groupby("query_id", sort=False)
    }


def source_texts(wanted: set[int]) -> dict[int, str]:
    found: dict[int, str] = {}
    paths = sorted((INDEX / "text").glob("part_*.parquet"))
    for number, path in enumerate(paths, 1):
        missing = wanted.difference(found)
        if not missing:
            break
        frame = pd.read_parquet(path, columns=["doc_id", "text"])
        subset = frame[frame.doc_id.isin(missing)]
        found.update((int(row.doc_id), str(row.text)) for row in subset.itertuples(index=False))
        if number % 15 == 0:
            print(f"source lookup {len(found):,}/{len(wanted):,}", flush=True)
    missing = wanted.difference(found)
    if missing:
        raise RuntimeError(f"Missing source text for {len(missing):,} selected documents")
    if any(not text.strip() for text in found.values()):
        raise RuntimeError("Empty source text among selected parents")
    return found


def build(output: Path, manifest_path: Path) -> dict:
    if any(p.exists() for p in (output, manifest_path, output.with_suffix(".json"),
                                 output.with_name(output.stem + ".building.zip"))):
        raise FileExistsError("Refusing to overwrite candidate, manifest, or staging files")
    qwen = load_rank_maps(QWEN_DOCS, "qwen4b_rank")
    bge = load_rank_maps(BGE_DOCS, "bge_rank")
    if set(qwen) != set(bge) or len(qwen) != QUERY_COUNT:
        raise RuntimeError("Qwen/BGE query coverage mismatch")
    official = set(pd.read_parquet(CORPUS, columns=["id"]).id.astype(int))

    # Candidate set is current submitted docs without any source chunk in the
    # original 130,108-chunk reference. Qwen/BGE ranks only choose among it.
    selected: dict[int, list[int]] = {}
    original_owners: dict[int, set[int]] = {}
    docs_by_query: dict[int, list[int]] = {}
    row_docs: dict[int, set[int]] = {}
    for row in read_rows(SOURCE):
        qid = int(row["id"])
        docs = list(map(int, row["relevant_docs"]))
        owners = {int(chunk["doc_id"]) for chunk in row["relevant_chunks"]}
        if not set(docs).issubset(official):
            raise RuntimeError(f"Source docs outside official corpus at query {qid}")
        docs_by_query[qid] = docs
        original_owners[qid] = owners
        row_docs[qid] = set(docs)
        eligible = [d for d in docs if d not in owners]
        qr, br = qwen[qid], bge[qid]
        fused = sorted(eligible, key=lambda d: (
            1.0 / (RRF_K + qr[d]) + 1.0 / (RRF_K + br[d]),
            -qr[d], -br[d], -d,
        ), reverse=True)
        selected[qid] = fused[:MAX_PER_QUERY]
    if len(selected) != QUERY_COUNT:
        raise RuntimeError("Expected 1,200 source queries")
    additions = {doc for values in selected.values() for doc in values}
    if not additions or not additions.issubset(official):
        raise RuntimeError("No eligible additions or additions outside official corpus")
    texts = source_texts(additions)
    lengths = [len(texts[d]) for d in additions]

    stage = output.with_suffix(".json")
    rows = chunks = added = 0
    try:
        with stage.open("x", encoding="utf-8", newline="\n") as dest:
            dest.write("[")
            for row in read_rows(CONTROL):
                qid = int(row["id"])
                if set(map(int, row["relevant_docs"])) != row_docs[qid]:
                    raise RuntimeError(f"Current docs differ from frozen source at query {qid}")
                existing = {(int(c["doc_id"]), str(c["chunk_text"])) for c in row["relevant_chunks"]}
                for doc_id in selected[qid]:
                    text = texts[doc_id]
                    if doc_id in original_owners[qid] or (doc_id, text) in existing:
                        raise RuntimeError(f"Addition is not missing/unique for query {qid}")
                    row["relevant_chunks"].append({"doc_id": doc_id, "chunk_text": text})
                    existing.add((doc_id, text))
                    added += 1
                if rows:
                    dest.write(",\n")
                dest.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
                rows += 1
                chunks += len(row["relevant_chunks"])
            dest.write("\n]\n")
        if rows != QUERY_COUNT or added != sum(map(len, selected.values())):
            raise RuntimeError("Output coverage/addition count mismatch")
        staged_zip = package_lzma64(stage, output)
        if staged_zip.stat().st_size >= 100_000_000:
            raise RuntimeError(f"Candidate exceeds decimal 100 MB: {staged_zip.stat().st_size}")
        with zipfile.ZipFile(staged_zip) as archive:
            if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
                raise RuntimeError("Candidate ZIP layout/CRC failure")
            with archive.open("submission.json") as packed, stage.open("rb") as raw:
                payload_hash = sha256_stream(packed)
                if payload_hash != sha256_stream(raw):
                    raise RuntimeError("Packaged output differs from source JSON")
        os.replace(staged_zip, output)
    except Exception:
        output.unlink(missing_ok=True)
        output.with_name(output.stem + ".building.zip").unlink(missing_ok=True)
        raise
    finally:
        stage.unlink(missing_ok=True)

    manifest = {
        "schema_version": "final_sprint_exp142_v1",
        "status": "exploratory_btc_trial_pending",
        "policy": {"kind": "append_full_source_parents_for_missing_original_chunk_owners",
                   "max_per_query": MAX_PER_QUERY, "rank_fusion": "equal-weight Qwen4B/BGE RRF",
                   "rrf_k": RRF_K, "base_chunk_filter": "Exp139 keep90"},
        "queries": rows, "docs_unchanged": True, "chunks": chunks,
        "additions": added, "selected_doc_count": len(additions),
        "parent_chars": {"min": min(lengths), "median": sorted(lengths)[len(lengths)//2],
                         "max": max(lengths), "total": sum(lengths)},
        "control_zip_sha256": sha256_file(CONTROL),
        "rank_cache_sha256": {"qwen4b": sha256_file(QWEN_DOCS), "bge": sha256_file(BGE_DOCS)},
        "submission_zip_sha256": sha256_file(output), "submission_zip_bytes": output.stat().st_size,
        "source_verbatim": True,
        "warning": "RRF rank is not relevance proof; only BTC score can evaluate this treatment.",
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "submission_exp142_missing_owners_add5.zip")
    parser.add_argument("--manifest", type=Path, default=ROOT / "submission_exp142_missing_owners_add5_manifest.json")
    args = parser.parse_args()
    print(json.dumps(build(args.output, args.manifest), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
