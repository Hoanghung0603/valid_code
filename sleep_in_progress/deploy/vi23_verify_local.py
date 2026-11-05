"""Verify final ZIPs locally, retaining a machine-readable report."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", required=True)
    args = parser.parse_args()
    base = Path(args.dir)
    reports = []
    for name in ["submission_vi23.zip", "submission_newonly30.zip"]:
        path = base / name
        with zipfile.ZipFile(path) as archive:
            entries = archive.infolist()
            assert len(entries) == 1
            entry = entries[0]
            assert not entry.is_dir() and "/" not in entry.filename and "\\" not in entry.filename
            assert entry.filename.endswith(".json") and entry.compress_type == zipfile.ZIP_LZMA
            rows = json.loads(archive.read(entry))
        assert len(rows) == 1200 and len({row["id"] for row in rows}) == 1200
        expected = {json.loads(line)["id"] for line in
                    (Path(__file__).parents[1] / "queries_test.jsonl").read_text(encoding="utf-8").splitlines()
                    if line.strip()}
        assert {row["id"] for row in rows} == expected
        bad = sum(chunk["doc_id"] not in set(row["relevant_docs"])
                  for row in rows for chunk in row["relevant_chunks"])
        assert bad == 0
        assert path.stat().st_size <= 93_000_000
        report = {"file": name, "bytes": path.stat().st_size,
                  "sha256": hashlib.file_digest(path.open("rb"), "sha256").hexdigest(),
                  "entry": entry.filename, "compression": "ZIP_LZMA", "queries": len(rows),
                  "chunks": sum(len(row["relevant_chunks"]) for row in rows),
                  "chunks_outside_docs": bad, "ok": True}
        reports.append(report)
    output = json.dumps(reports, ensure_ascii=False, indent=2)
    (base / "local_zip_report.json").write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
