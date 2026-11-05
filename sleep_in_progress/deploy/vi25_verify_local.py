"""Verify downloaded vi25 ZIPs and write a compact local evidence report."""
import hashlib
import json
from pathlib import Path
import zipfile

LIMITS = {"submission_vi25_z93.zip": 93_000_000,
          "submission_vi25_z99.zip": 99_000_000,
          "submission_vi25_newonly30.zip": 99_000_000}


def main():
    base = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + r"\diag25")
    remote = {}
    fp = base / "zip_fingerprints.json"
    if fp.exists():
        remote = {Path(r["path"]).name: r["sha256"] for r in json.loads(fp.read_text(encoding="utf-8"))}
    report = []
    for name, limit in LIMITS.items():
        path = base / name
        if not path.exists():
            continue
        assert path.stat().st_size <= limit, (name, path.stat().st_size, limit)
        with zipfile.ZipFile(path) as z:
            info, = z.infolist()
            assert "/" not in info.filename and "\\" not in info.filename
            assert info.compress_type == zipfile.ZIP_LZMA
            with z.open(info) as f:
                data = json.load(f)
        assert len(data) == len({str(r["id"]) for r in data}) == 1200
        assert {int(r["id"]) for r in data} == set(range(1, 1201))
        bad = sum(str(c["doc_id"]) not in {str(d) for d in r["relevant_docs"]}
                  for r in data for c in r["relevant_chunks"])
        assert bad == 0
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in remote:
            assert remote[name] == sha, (name, remote[name], sha)
        report.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha,
                       "sha256_giong_remote": remote.get(name) == sha, "queries": len(data),
                       "chunk_doc_mismatch": bad, "method": "ZIP_LZMA", "entry": info.filename,
                       "duoi_tran": path.stat().st_size <= limit,
                       "docs_per_q": sum(len(r["relevant_docs"]) for r in data) / len(data),
                       "chunks_per_q": sum(len(r["relevant_chunks"]) for r in data) / len(data)})
    assert report
    (base / "local_verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
