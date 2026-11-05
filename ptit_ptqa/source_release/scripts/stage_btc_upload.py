"""Stage exactly the required Stage 3 deliverables in one flat upload folder.

Hardlinks keep the 25 GB of index volumes from being copied on the same NTFS
volume. The links are ordinary files for upload; changing one changes its peer.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "deliverables/stage3_20261104/artifact_manifest.json"
TARGET = ROOT / "deliverables/STAGE3_UPLOAD_20261104"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    records = [item for item in manifest["artifacts"] if item["required"]]
    sources: list[tuple[Path, Path, dict]] = []
    seen_names: set[str] = set()
    for record in records:
        name = record["name"]
        source = (ROOT / record["local_path"]).resolve(strict=True)
        if not source.is_relative_to(ROOT.resolve()) or not source.is_file():
            raise RuntimeError(f"Unsafe or missing source: {name}")
        if record["status"] != "ready" or source.stat().st_size != record["bytes"]:
            raise RuntimeError(f"Manifest status/size mismatch: {name}")
        if name in seen_names or name != source.name:
            raise RuntimeError(f"Duplicate or renamed artifact: {name}")
        seen_names.add(name)
        sources.append((source, TARGET / name, record))
    sources.append((MANIFEST, TARGET / MANIFEST.name, {"name": MANIFEST.name}))
    if len(sources) != 26:
        raise RuntimeError(f"Unexpected upload file count: {len(sources)}")
    TARGET.mkdir(parents=True, exist_ok=True)
    linked = 0
    total_bytes = 0
    for source, dest, record in sources:
        if "sha256" in record and digest(source) != record["sha256"]:
            raise RuntimeError(f"Source SHA256 differs from manifest: {source}")
        if dest.exists():
            if not os.path.samefile(source, dest):
                raise FileExistsError(f"Unexpected file already in upload folder: {dest}")
        else:
            os.link(source, dest)
            linked += 1
        if not os.path.samefile(source, dest) or dest.stat().st_size != source.stat().st_size:
            raise RuntimeError(f"Hardlink verification failed: {dest}")
        total_bytes += dest.stat().st_size
    actual = {p.name for p in TARGET.iterdir() if p.is_file()}
    expected = {dest.name for _, dest, _ in sources}
    if actual != expected:
        raise RuntimeError(f"Upload folder mismatch: missing={expected-actual}, extra={actual-expected}")
    print(json.dumps({"upload_folder": str(TARGET), "files": len(sources),
                      "new_hardlinks": linked, "logical_bytes": total_bytes,
                      "extra_disk_copy_bytes": 0, "manifest": str(MANIFEST)}, indent=2))


if __name__ == "__main__":
    main()
