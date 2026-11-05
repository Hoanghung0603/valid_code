"""Check that two submissions hold the same docs and chunks for every query (query order may differ).

  python scripts/compare_submissions.py data/runs/sub-long-kproxy-nproxy-v1/submission.json \
         data/reference/sub-long-kproxy-nproxy-v1/submission.json
A path whose JSON is gone is read from the zip in the same folder (medret.submit.iter_submission), so the second
argument is the submitted zip. Entries are compared through a SHA-256 of their canonical JSON (doc order and chunk
order included), so neither file is held in memory. The packing step reorders queries (overlap_order), which is
why the submitted file differs from the fused_sub.py output byte for byte while every entry is identical.
Exit code 1 if any query differs.
"""

from __future__ import annotations

import hashlib
import json
import sys

from medret.submit import iter_submission


def digests(path: str) -> dict[int, str]:
    return {int(x["id"]): hashlib.sha256(json.dumps(x, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
            for x in iter_submission(path)}


def main() -> None:
    a, b = digests(sys.argv[1]), digests(sys.argv[2])
    diff = sorted(q for q in a.keys() | b.keys() if a.get(q) != b.get(q))
    print(f"{len(a)} vs {len(b)} queries; {len(diff)} differ" + (f" (first: {diff[:10]})" if diff else ": identical"))
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
