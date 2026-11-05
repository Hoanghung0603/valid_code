"""Check a gold-set extraction run against NT's approval (reviews/gold_gate.json).

TV never receives the gold answers, so the notebook cannot score itself.
Instead NT scores a run, records a digest of its outputs, and the notebook
proves it reproduces exactly that run before cleaning the 40K sample.

The digest covers what scoring depends on: text, block types and offsets,
quality flags. JSON formatting does not matter.

Usage:
    python eval/gold_gate.py reviews/gold_gate.json \
        gold1=<run>/extract_gold1_latest gold2=... gold3=...
In a notebook:
    from gold_gate import check_gate
    check_gate(gate_json, {"gold1": dir1, ...}, postfilter_sha1, hints_sha1)
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

EXTRACTOR = "custom_cleaner_v3"


def outputs_digest(run_dir: Path) -> tuple[str, int]:
    """sha1 over every <id>.json of the extractor, ordered by id."""
    files = sorted((Path(run_dir) / EXTRACTOR).glob("*.json"), key=lambda p: int(p.stem))
    h = hashlib.sha1()
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        blocks = ",".join(f"{b.get('type')}:{b.get('char_start')}:{b.get('char_end')}"
                          for b in doc.get("blocks") or [])
        flags = ",".join(sorted(doc.get("quality_flags") or []))
        text_sha1 = hashlib.sha1((doc.get("text") or "").encode("utf-8")).hexdigest()
        h.update(f"{f.stem}\t{text_sha1}\t{blocks}\t{flags}\n".encode("utf-8"))
    return h.hexdigest(), len(files)


def check_gate(gate_path, run_dirs: dict, postfilter_sha1: str, hints_sha1: str) -> list[str]:
    """Return a list of problems; empty means the run matches NT's approved run."""
    gate = json.loads(Path(gate_path).read_text(encoding="utf-8"))
    problems = []
    if postfilter_sha1 != gate["postfilter_sha1"]:
        problems.append(f"postfilter_sha1 {postfilter_sha1} != {gate['postfilter_sha1']}")
    if hints_sha1 != gate["container_hints_sha1"]:
        problems.append(f"container_hints_sha1 {hints_sha1} != {gate['container_hints_sha1']}")
    for name, approved in gate["gold_sets"].items():
        if name not in run_dirs:
            problems.append(f"{name}: no run directory given")
            continue
        digest, n = outputs_digest(run_dirs[name])
        if n != approved["docs"] or digest != approved["outputs_digest"]:
            problems.append(f"{name}: {n} docs, digest {digest} "
                            f"!= approved {approved['docs']} docs, {approved['outputs_digest']}")
    return problems


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    approved = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))["gold_sets"]
    runs = dict(a.split("=", 1) for a in sys.argv[2:])
    for name, d in runs.items():
        digest, n = outputs_digest(Path(d))
        same = name in approved and approved[name]["outputs_digest"] == digest
        print(f"{name}: docs={n} outputs_digest={digest} {'MATCH' if same else 'DIFFERENT'}")


if __name__ == "__main__":
    main()
