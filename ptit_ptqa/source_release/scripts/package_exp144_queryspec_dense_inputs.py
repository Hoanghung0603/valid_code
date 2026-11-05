"""Package QuerySpec BM25 candidates and the Exp143 control for Kaggle."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

try:
    from build_final_sprint_exp142_missing_parents import read_rows
except ModuleNotFoundError:
    from scripts.build_final_sprint_exp142_missing_parents import read_rows

ROOT = Path(__file__).resolve().parents[1]
SPECS = ROOT / "data/query_specs/queryspec_production_v1/query_specs.parquet"
ATOMS = ROOT / "data/query_specs/queryspec_production_v1/evidence_atoms.parquet"
CANDIDATES = ROOT / "data/indices/bm25_document/queryspec_candidates_v1.parquet"
QUERIES = ROOT / "data/raw/btc_provided/query.parquet"
CONTROL = ROOT / "submission_exp143_p2a_long_chunks_add3.zip"
AUDIT = ROOT / "data/audits/queryspec_candidates_v1/audit.json"
OUTPUT = ROOT / "data/kaggle_inputs/exp144_queryspec_dense_input.zip"
MANIFEST = ROOT / "data/kaggle_inputs/exp144_queryspec_dense_input_manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace only the two deterministic Exp144 package outputs")
    args = parser.parse_args()
    required = [SPECS, ATOMS, CANDIDATES, QUERIES, CONTROL, AUDIT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing Exp144 inputs: {missing}")
    if (OUTPUT.exists() or MANIFEST.exists()) and not args.force:
        raise FileExistsError("Refusing to overwrite Exp144 Kaggle package")
    specs = pd.read_parquet(SPECS, columns=["query_id", "status", "expected_min_docs", "expected_max_docs"])
    candidates = pd.read_parquet(CANDIDATES, columns=["query_id", "rank", "doc_id"])
    queries = pd.read_parquet(QUERIES, columns=["id", "query"])
    if len(specs) != 1200 or specs.query_id.nunique() != 1200:
        raise RuntimeError("QuerySpec production coverage is not 1,200")
    if len(queries) != 1200 or queries.id.nunique() != 1200:
        raise RuntimeError("Official query coverage is not 1,200")
    if candidates.query_id.nunique() != 1200 or candidates.duplicated(["query_id", "doc_id"]).any():
        raise RuntimeError("Candidate coverage/uniqueness failed")
    with zipfile.ZipFile(CONTROL) as archive:
        if archive.namelist() != ["submission.json"] or archive.testzip() is not None:
            raise RuntimeError("Exp143 control ZIP invalid")
    control_rows = []
    for row in read_rows(CONTROL):
        query_id = int(row["id"])
        control_rows.extend(
            {"query_id": query_id, "rank": rank, "doc_id": int(doc_id)}
            for rank, doc_id in enumerate(row["relevant_docs"], 1)
        )
    control_docs = pd.DataFrame(control_rows)
    if control_docs.query_id.nunique() != 1200 or control_docs.duplicated(["query_id", "doc_id"]).any():
        raise RuntimeError("Exp143 control docs coverage/uniqueness failed")
    control_buffer = io.BytesIO()
    control_docs.to_parquet(control_buffer, index=False, compression="zstd")
    control_payload = control_buffer.getvalue()
    file_payloads = {
        "query_specs.parquet": SPECS,
        "evidence_atoms.parquet": ATOMS,
        "queryspec_candidates.parquet": CANDIDATES,
        "query.parquet": QUERIES,
        "candidate_audit.json": AUDIT,
    }
    manifest = {
        "schema_version": "exp144_queryspec_dense_input_v1",
        "purpose": "QuerySpec/BM25 candidate expansion followed by original-query exact BGE-M3 reranking",
        "queries": 1200,
        "candidate_rows": len(candidates),
        "candidate_count": candidates.groupby("query_id").size().describe().to_dict(),
        "queryspec_status": specs.status.value_counts().to_dict(),
        "control": "Exp143 docs in rank-preserving Parquet; chunks remain local",
        "control_submission_sha256": sha256(CONTROL),
        "files": {
            **{name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
               for name, path in file_payloads.items()},
            "exp143_control_docs.parquet": {
                "bytes": len(control_payload), "sha256": sha256_bytes(control_payload),
                "rows": len(control_docs),
            },
        },
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name, path in file_payloads.items():
            archive.write(path, name)
        archive.writestr("exp143_control_docs.parquet", control_payload)
        archive.write(MANIFEST, "input_manifest.json")
    with zipfile.ZipFile(OUTPUT) as archive:
        expected = set(file_payloads) | {"exp143_control_docs.parquet", "input_manifest.json"}
        if archive.testzip() is not None or set(archive.namelist()) != expected:
            raise RuntimeError("Exp144 package CRC/layout failed")
    manifest["package"] = {"bytes": OUTPUT.stat().st_size, "sha256": sha256(OUTPUT)}
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
