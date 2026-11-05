"""Verify every rescued source payload and repair extraction of our new rows.

Only IDs introduced by publisher_rescue_20261019 may be updated. Existing rows
must still equal our saved extraction, otherwise concurrent edits are rejected.
"""

from datetime import datetime
import gzip
import hashlib
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import audit_local_recovered_final as audit
from recover_publisher_frontends import text_from_html


def main():
    original = audit.DATA / "audits/publisher_rescue_20261019"
    out = original / "text_verification"
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((original / "applied_summary.json").read_text(encoding="utf-8"))
    own_ids = set(manifest["new_unembedded_ids"])
    hashes = {name: audit.digest(path) for name, path in audit.REGISTRIES.items()}
    frames = {name: pd.read_parquet(path, columns=["id", "url"] if name == "embedded" else None)
              for name, path in audit.REGISTRIES.items()}
    official = pd.read_parquet(audit.OFFICIAL, columns=["id", "url"])
    audit.validate_partition(official, frames)
    old_u = frames["unembedded"].set_index("id", drop=False)
    updated_u = old_u.copy()
    checked_ids = set()
    changed_ids = []
    source_hashes = {}
    ledger_updates = []
    source_dir = audit.RECOVERED / "pharmacity_api_20261019"
    for path in sorted(source_dir.glob("api_part_*.parquet")):
        frame = pd.read_parquet(path)
        frame_changed = False
        for source, group in frame[frame["status"].eq("success")].groupby("source_file"):
            raw = gzip.decompress((audit.ROOT / source).read_bytes())
            digest = hashlib.sha256(raw).hexdigest()
            assert set(group["response_sha256"]) == {digest}
            source_hashes[source] = digest
            by_url = {post["link"]: post for post in json.loads(raw)}
            for index, row in group.iterrows():
                row_id = int(row["id"])
                assert row_id in own_ids and row_id not in checked_ids
                post = by_url[row["url"]]
                assert post["status"] == "publish" and post["id"] == row["wordpress_id"]
                body = text_from_html(post["content"]["rendered"])
                assert body.strip(), f"No source body for {row_id}"
                expected = text_from_html(post["title"]["rendered"] + "\n" + post["content"]["rendered"])
                assert old_u.at[row_id, "url"] == row["url"]
                assert old_u.at[row_id, "text"] in (row["text"], expected), f"Concurrent text change: {row_id}"
                if expected != old_u.at[row_id, "text"]:
                    updated_u.at[row_id, "text"] = expected
                    changed_ids.append(row_id)
                if expected != row["text"]:
                    frame.at[index, "text"] = expected
                    frame_changed = True
                checked_ids.add(row_id)
        if frame_changed:
            ledger_updates.append((path, frame))
    assert checked_ids == own_ids
    updated = {name: frame.copy() for name, frame in frames.items()}
    updated["unembedded"] = updated_u.reset_index(drop=True)
    audit.validate_partition(official, updated)
    changed = audit.apply_changes(frames, updated, hashes, out) if changed_ids else []
    for path, frame in ledger_updates:
        backup = out / (path.name + ".backup.parquet")
        if backup.exists():
            raise RuntimeError("Earlier ledger backup exists; refusing overwrite")
        import shutil
        shutil.copy2(path, backup)
        temp = path.with_suffix(".verification.tmp")
        frame.to_parquet(temp, index=False)
        temp.replace(path)
    result = {
        "checked_at": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(),
        "exact_source_records_verified": len(checked_ids), "all_source_bodies_nonempty": True,
        "text_repairs": sorted(changed_ids), "changed_registries": changed,
        "input_sha256": hashes,
        "output_sha256": {name: audit.digest(path) for name, path in audit.REGISTRIES.items()},
        "source_payloads": source_hashes,
        "partition_invariant": "PASS",
    }
    (out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "source_payloads"}, indent=2))


if __name__ == "__main__":
    main()
