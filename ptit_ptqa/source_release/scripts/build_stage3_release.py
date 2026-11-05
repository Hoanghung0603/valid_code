"""Build the curated Stage 3 source release and outer artifact manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables/stage3_20261104"

SOURCE_ROOTS = ("configs", "src", "scripts", "notebooks", "demo", "tests", "design-system")
SOURCE_SUFFIXES = {".py", ".ps1", ".ipynb", ".json", ".md", ".txt", ".html", ".css", ".js", ".yml", ".yaml", ".toml"}
SOURCE_FILES = ["requirements.txt", "environment.yml", "pytest.ini"]
for root_name in SOURCE_ROOTS:
    for source_path in sorted((ROOT / root_name).rglob("*")):
        if (source_path.is_file() and source_path.suffix.lower() in SOURCE_SUFFIXES
                and not any(part in {"__pycache__", ".ipynb_checkpoints", "node_modules"}
                            for part in source_path.relative_to(ROOT).parts)
                and source_path.name not in {".env", ".env.local"}):
            SOURCE_FILES.append(source_path.relative_to(ROOT).as_posix())

SECRET_PATTERNS = [
    re.compile(rb"OPENROUTER_API_KEY\s*=\s*[^\s]+", re.I),
    re.compile(rb"CF_TOKEN\s*=\s*[^\s]+", re.I),
    re.compile(rb"hf_[A-Za-z0-9]{20,}"),
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_safe(path: Path, payload: bytes) -> None:
    if path.name in {".env", ".env.local"}:
        raise ValueError(f"Credential file cannot enter source release: {path}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(payload):
            raise ValueError(f"Potential credential in {path}: {pattern.pattern!r}")


def build_source_zip(output: Path) -> dict:
    entries: list[dict] = []
    payloads: list[tuple[str, bytes]] = []
    for archive_name, path in [
        ("README.md", DELIVERABLES / "README.md"),
        ("requirements-release.txt", DELIVERABLES / "requirements-release.txt"),
    ]:
        payload = path.read_bytes()
        assert_safe(path, payload)
        payloads.append((archive_name, payload))
    for relative in SOURCE_FILES:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = path.read_bytes()
        assert_safe(path, payload)
        payloads.append((relative.replace("\\", "/"), payload))
    for name, payload in payloads:
        entries.append({"path": name, "bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest()})
    manifest = {
        "schema_version": "r2ai_source_manifest_v1",
        "release_id": "stage3-exp143-20261103",
        "source_commit": "0cee33e74e75bb2b849f7814f611b68e173165e1",
        "source_state": "snapshot of working tree; may contain uncommitted changes",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection": "curated release inventory; not git archive",
        "files": entries,
    }
    payloads.append(("source_manifest.json",
                     (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode("utf-8")))
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(".building.zip")
    staging.unlink(missing_ok=True)
    with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, payload in payloads:
            info = zipfile.ZipInfo(name, date_time=(2026, 11, 3, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, payload)
    if output.exists():
        output.unlink()
    staging.replace(output)
    with zipfile.ZipFile(output) as archive:
        if archive.testzip() is not None:
            raise ValueError("Source archive CRC validation failed")
    return {"path": output.name, "bytes": output.stat().st_size,
            "sha256": sha256_file(output), "files": len(payloads), "secret_scan": "pass"}


def artifact_record(path: Path, role: str, required: bool = True) -> dict:
    record = {"name": path.name, "role": role, "required": required,
              "local_path": str(path.relative_to(ROOT)).replace("\\", "/")}
    if path.is_file():
        record.update(status="ready", bytes=path.stat().st_size, sha256=sha256_file(path))
    elif path.is_dir():
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        record.update(status="directory_requires_packaging", bytes=size)
    else:
        record.update(status="missing")
    return record


def build_artifact_manifest(source_result: dict) -> dict:
    artifacts = [
        artifact_record(DELIVERABLES / "technical_report.pdf", "BTC technical report"),
        artifact_record(DELIVERABLES / "presentation.pptx", "Editable presentation"),
        artifact_record(DELIVERABLES / "presentation.pdf", "Rendered presentation"),
        artifact_record(DELIVERABLES / "source_release.zip", "Curated source/config/test/UI"),
        artifact_record(DELIVERABLES / "README.md", "BTC installation and replay instructions"),
        artifact_record(DELIVERABLES / "model_registry.json", "Model IDs, revisions, roles and official checkpoint links"),
        artifact_record(DELIVERABLES / "UPLOAD_INVENTORY.md", "Data and source upload inventory"),
        artifact_record(DELIVERABLES / "access_checklist.md", "Sharing and submission acceptance checklist"),
        artifact_record(DELIVERABLES / "clean_room_validation.md", "Independent release tests and limitations"),
        artifact_record(ROOT / "data/raw/btc_provided/query.parquet", "Official queries; included in text_snapshot volumes", required=False),
        artifact_record(ROOT / "data/raw/btc_provided/links_corpus_vi_en_zh.parquet", "Official corpus ID/URL mapping; included in text_snapshot volumes", required=False),
        artifact_record(ROOT / "data/embedded_samples.parquet", "Clean text snapshot; included in text_snapshot volumes", required=False),
        *[artifact_record(p, "Active BGE-M3 index volume") for p in sorted(DELIVERABLES.glob("active_index.7z.*"))],
        *[artifact_record(p, "Text snapshot and BTC query/corpus volume") for p in sorted(DELIVERABLES.glob("text_snapshot.7z.*"))],
        *[artifact_record(p, "Secondary Qwen index volume") for p in sorted(DELIVERABLES.glob("qwen_secondary_index.7z.*"))],
        artifact_record(DELIVERABLES / "exp143_rebuild_inputs.7z", "Cached upstream inputs to regenerate Exp143 without replay ZIP"),
        artifact_record(ROOT / "deliverables/_optional_private_variants/private_variants_inputs.7z", "QuerySpec/BM25 inputs for Private02-05", required=False),
        *[artifact_record(p, "Exploratory private submission; upload only if submitted", required=False)
          for p in sorted(ROOT.glob("submission_private0[2345]_*.zip"))],
        *[artifact_record(p, "Exploratory private submission manifest", required=False)
          for p in sorted(ROOT.glob("submission_private0[2345]_*_manifest.json"))],
        artifact_record(ROOT / "data/cache/release_exp143/replay.sqlite", "Compressed UI replay cache", required=False),
        artifact_record(ROOT / "data/reference/concept_graph_v1.sqlite", "Experimental UI concept graph; not in champion", required=False),
        artifact_record(ROOT / "data/reference/concept_graph_v1.manifest.json", "Experimental graph provenance", required=False),
        artifact_record(ROOT / "submission_private01_exp143_replay.zip", "Audited champion submission"),
    ]
    manifest = {
        "schema_version": "r2ai_artifact_manifest_v1",
        "release_id": "stage3-exp143-20261103",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_release": source_result, "artifacts": artifacts,
        "drive_root": None,
        "drive_status": "USER_ACTION_REQUIRED: upload, fill URL, and verify in an unauthenticated session",
        "exclusions": [".env", "model download caches", "optimizer states", "backup indexes", "temporary experiment files"],
    }
    path = DELIVERABLES / "artifact_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    output = DELIVERABLES / "source_release.zip"
    source = ({"path": output.name, "bytes": output.stat().st_size,
               "sha256": sha256_file(output)}
              if args.manifest_only and output.exists() else build_source_zip(output))
    manifest = build_artifact_manifest(source)
    print(json.dumps({"source_release": source,
                      "artifact_status": {item["name"]: item["status"] for item in manifest["artifacts"]}},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
