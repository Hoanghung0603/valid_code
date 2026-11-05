import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_local_recovered_final import validate_partition, verified_dead
from scripts import audit_local_recovered_final as audit


def test_blocks_redirects_and_api_misses_never_authorize_exclusion():
    rows = []
    for row_id, status, blocked, kind, final in (
        (1, 403, True, "origin", "https://a/1"),
        (2, 404, False, "origin", "https://a/notfound"),
        (3, 404, False, "api", "https://a/3"),
    ):
        for _ in range(2):
            rows.append(dict(id=row_id, url=f"https://a/{row_id}",
                             status=status, blocked=blocked, kind=kind, final_url=final))
    assert verified_dead(pd.DataFrame(rows), set()) == set()


def test_success_precedes_even_two_source_404s_and_single_probe_is_insufficient():
    rows = [dict(id=i, url=f"https://a/{i}", final_url=f"https://a/{i}",
                 status=404, blocked=False, kind="origin") for i in (1, 1, 2, 3, 3)]
    assert verified_dead(pd.DataFrame(rows), {1}) == {3}


def test_official_id_with_wrong_url_is_not_safe_to_write():
    official = pd.DataFrame({"id": [1, 2], "url": ["https://a/1", "https://a/2"]})
    frames = {"embedded": official.iloc[:1].copy(), "remaining": official.iloc[1:].copy()}
    validate_partition(official, frames)
    frames["remaining"].loc[:, "url"] = "https://wrong/2"
    with pytest.raises(ValueError, match="ID/URL"):
        validate_partition(official, frames)


def test_partition_rejects_overlap_and_missing_rows():
    official = pd.DataFrame({"id": [1, 2], "url": ["https://a/1", "https://a/2"]})
    with pytest.raises(ValueError, match="overlap"):
        validate_partition(official, {"embedded": official, "remaining": official.iloc[:1]})
    with pytest.raises(ValueError, match="exact partition"):
        validate_partition(official, {"embedded": official.iloc[:1]})


def test_staged_parquet_update_preserves_backups_and_embedded_file(tmp_path, monkeypatch):
    official = pd.DataFrame({"id": [1, 2, 3, 4], "url": [f"https://a/{i}" for i in range(1, 5)]})
    frames = {
        "embedded": official.iloc[[0]].reset_index(drop=True),
        "unembedded": official.iloc[[1]].assign(text="old text").reset_index(drop=True),
        "remaining": official.iloc[[2]].reset_index(drop=True),
        "excluded": official.iloc[[3]].assign(reason="previous exclusion").reset_index(drop=True),
    }
    paths = {name: tmp_path / (name + ".parquet") for name in frames}
    for name, frame in frames.items():
        frame.to_parquet(paths[name], index=False)
    hashes = {name: audit.digest(path) for name, path in paths.items()}
    updated = {name: frame.copy() for name, frame in frames.items()}
    updated["unembedded"] = pd.concat([
        frames["unembedded"], official.iloc[[2]].assign(text="new recovered text")
    ], ignore_index=True)
    updated["unembedded"].columns = pd.Index(["id", "url", "text"], dtype="str")
    updated["remaining"] = frames["remaining"].iloc[:0].copy()
    validate_partition(official, updated)
    monkeypatch.setattr(audit, "REGISTRIES", paths)
    changed = audit.apply_changes(frames, updated, hashes, tmp_path)
    assert set(changed) == {"unembedded", "remaining"}
    assert audit.digest(paths["embedded"]) == hashes["embedded"]
    for name in changed:
        assert audit.digest(tmp_path / "backups" / paths[name].name) == hashes[name]
    restored = {name: pd.read_parquet(path) for name, path in paths.items()}
    validate_partition(official, restored)
    assert restored["unembedded"]["id"].tolist() == [2, 3]
