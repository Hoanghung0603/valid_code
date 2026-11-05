"""Audit recovery history and reconcile only proven successes/dead source URLs.

Dry run by default. Network probes are recorded per URL; domain samples never
authorize deleting other URLs. Transient blocks and secondary-index misses
remain retryable even when the original crawler called them 'dropped'.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
from urllib.parse import urlparse
from zoneinfo import ZoneInfo
from uuid import uuid4

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RECOVERED = DATA / "raw/corpus_text/local_recovered"
OFFICIAL = DATA / "raw/btc_provided/links_corpus_vi_en_zh.parquet"
REGISTRIES = {
    "embedded": DATA / "embedded_samples.parquet",
    "unembedded": DATA / "unembedded_samples.parquet",
    "remaining": DATA / "remaining_urls.parquet",
    "excluded": DATA / "excluded_urls.parquet",
}


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def domain(url: str) -> str:
    return (urlparse(url).hostname or "").lower().removeprefix("www.")


def validate_partition(official: pd.DataFrame, frames: dict[str, pd.DataFrame]) -> None:
    if official["id"].duplicated().any() or official["url"].duplicated().any():
        raise ValueError("Official mapping is not unique")
    mapping = official.set_index("id")["url"]
    ids: set[int] = set()
    for name, frame in frames.items():
        if frame[["id", "url"]].isna().any().any():
            raise ValueError(f"{name}: null identity")
        if frame["id"].duplicated().any() or frame["url"].duplicated().any():
            raise ValueError(f"{name}: duplicate identity")
        if not frame["id"].map(mapping).eq(frame["url"]).all():
            raise ValueError(f"{name}: ID/URL differs from official mapping")
        current = set(frame["id"].astype(int))
        if current & ids:
            raise ValueError(f"{name}: registry overlap")
        ids.update(current)
    if ids != set(official["id"].astype(int)):
        raise ValueError("Registries are not an exact partition of official IDs")


def verified_dead(probes: pd.DataFrame, recovered_ids: set[int]) -> set[int]:
    """Require two fresh original-source probes with an exact URL match.

    JSON API misses, domain-wide failures, redirects and soft-404 heuristics
    are deliberately insufficient to remove an official URL.
    """
    if probes.empty:
        return set()
    source = probes[probes["kind"].eq("origin")].copy()
    result = set()
    for row_id, group in source.groupby("id"):
        row_id = int(row_id)
        if row_id in recovered_ids:
            continue
        if len(group) >= 2 and group["status"].isin([404, 410]).all():
            if group["final_url"].eq(group["url"]).all() and not group["blocked"].any():
                result.add(row_id)
    return result


def probe_url(row: dict) -> dict:
    from curl_cffi import requests
    from lxml import html

    record = {
        "id": int(row["id"]), "url": str(row["url"]), "kind": "origin",
        "checked_at": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(),
        "status": 0, "final_url": "", "blocked": False,
        "title": "", "body_sha256": "", "body_bytes": 0, "error": "",
    }
    try:
        response = requests.get(row["url"], timeout=16, impersonate="chrome124")
        record.update(status=response.status_code, final_url=response.url,
                      body_sha256=hashlib.sha256(response.content).hexdigest(),
                      body_bytes=len(response.content))
        low = response.text.lower()
        record["blocked"] = any(marker in low for marker in (
            "cookies must be enabled", "just a moment", "www error blocked diagnostic",
            "cf-browser-verification", "attention required", "access denied",
        )) or response.status_code in (203, 403, 429)
        try:
            tree = html.fromstring(response.content)
            record["title"] = tree.xpath("string(//title)").strip()[:200]
        except (ValueError, TypeError):
            pass
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"[:300]
    return record


def run_network_checks(remaining: pd.DataFrame, out: Path) -> pd.DataFrame:
    from curl_cffi import requests

    # All small non-PubMed domains; five evenly spaced samples of large domains.
    chosen = []
    for host, group in remaining.groupby(remaining["url"].map(domain)):
        limit = 3 if host == "pubmed.ncbi.nlm.nih.gov" else 5
        if len(group) <= 10 and host != "pubmed.ncbi.nlm.nih.gov":
            selected = group
        else:
            indexes = sorted({round(i * (len(group) - 1) / (limit - 1)) for i in range(limit)})
            selected = group.iloc[indexes]
        chosen.extend(selected[["id", "url"]].to_dict("records"))
    records = []
    with ThreadPoolExecutor(max_workers=6) as executor:
        for record in executor.map(probe_url, chosen):
            records.append(record)
            print(json.dumps({k: record[k] for k in ("id", "status", "blocked", "error")}), flush=True)
    # Reconfirm individual 404/410s only. A domain sample is never extrapolated.
    retry = [record for record in records if record["status"] in (404, 410)]
    with ThreadPoolExecutor(max_workers=3) as executor:
        records.extend(executor.map(probe_url, retry))
    probes = pd.DataFrame(records)
    probes.to_parquet(out / "origin_probes.parquet", index=False)

    pubmed = remaining[remaining["url"].map(domain).eq("pubmed.ncbi.nlm.nih.gov")]
    pmids = [re.search(r"/(\d+)(?:/|$)", url).group(1) for url in pubmed["url"]]
    api_checks = []
    for offset in range(0, len(pmids), 25):
        batch = pmids[offset:offset + 25]
        query = "SRC:MED AND (" + " OR ".join("EXT_ID:" + value for value in batch) + ")"
        check = {"pmids": batch, "query": query, "found_pmids": [], "error": ""}
        try:
            response = requests.get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                                    params={"query": query, "format": "json", "pageSize": 100}, timeout=20)
            response.raise_for_status()
            payload = response.json()
            if "hitCount" not in payload or "resultList" not in payload:
                raise ValueError("Unexpected Europe PMC schema")
            check["hit_count"] = payload["hitCount"]
            check["found_pmids"] = [str(item["id"]) for item in payload["resultList"].get("result", [])]
        except Exception as error:
            check["error"] = f"{type(error).__name__}: {error}"[:300]
        api_checks.append(check)
    (out / "europe_pmc_checks.json").write_text(json.dumps(api_checks, indent=2), encoding="utf-8")
    return probes


def apply_changes(frames: dict[str, pd.DataFrame], updated: dict[str, pd.DataFrame],
                  before_hashes: dict[str, str], out: Path) -> list[str]:
    changed = [name for name in ("unembedded", "remaining", "excluded")
               if not updated[name].equals(frames[name])]
    for name, path in REGISTRIES.items():
        if digest(path) != before_hashes[name]:
            raise RuntimeError(f"Concurrent change in {name}; refusing write")
    backups = out / "backups"
    backups.mkdir(exist_ok=True)
    staged = {}
    for name in changed:
        path = REGISTRIES[name]
        backup = backups / path.name
        if backup.exists():
            if digest(backup) != before_hashes[name]:
                raise RuntimeError(f"Refusing to overwrite an earlier backup: {backup}")
        else:
            shutil.copy2(path, backup)
        temp = path.with_name(path.name + f".final_audit.{uuid4().hex}.tmp")
        updated[name].to_parquet(temp, index=False, compression="snappy")
        # Arrow/Pandas can change string/index storage metadata on round trip.
        # Compare all values, column order and identity; do not require the
        # in-memory storage dtype to be byte-identical to its original.
        pd.testing.assert_frame_equal(pd.read_parquet(temp), updated[name].reset_index(drop=True),
                                      check_dtype=False, check_index_type=False, check_column_type=False)
        staged[name] = temp
    for name, path in REGISTRIES.items():
        if digest(path) != before_hashes[name]:
            raise RuntimeError(f"Concurrent change in {name} during staging; refusing commit")
    replaced = []
    try:
        for name in changed:
            os.replace(staged[name], REGISTRIES[name])
            replaced.append(name)
    except Exception:
        for name in replaced:
            shutil.copy2(backups / REGISTRIES[name].name, REGISTRIES[name])
        raise
    return changed


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--network", action="store_true")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.apply and (out / "applied_summary.json").exists():
        raise RuntimeError("This audit already committed; use a new output directory to preserve its evidence")
    before_hashes = {name: digest(path) for name, path in REGISTRIES.items()}
    official = pd.read_parquet(OFFICIAL, columns=["id", "url"])
    frames = {name: pd.read_parquet(path, columns=["id", "url"] if name == "embedded" else None)
              for name, path in REGISTRIES.items()}
    validate_partition(official, frames)
    history_parts = []
    for path in sorted(RECOVERED.rglob("*.parquet")):
        frame = pd.read_parquet(path)
        frame["ledger_file"] = str(path.relative_to(ROOT))
        if "source_file" not in frame:
            frame["source_file"] = ""
        frame["file_mtime_ns"] = path.stat().st_mtime_ns
        history_parts.append(frame)
    history = pd.concat(history_parts, ignore_index=True)
    official_url = official.set_index("id")["url"]
    if not history["id"].map(official_url).eq(history["url"]).all():
        raise ValueError("Recovered history contains unofficial ID/URL pairs")
    successes = history[history["status"].eq("success") & history["text"].notna()].copy()
    # A guessed publisher migration based only on slug/canonical is provisional.
    # Exact WordPress links or original-source recovery are required for sync.
    if "identity_evidence" in successes:
        successes = successes[~successes["identity_evidence"].fillna("").str.startswith("same_publisher_")].copy()
    successes = successes[successes["text"].astype(str).str.strip().str.len().gt(0)]
    successes["text_chars"] = successes["text"].str.len()
    successes = successes.sort_values(["text_chars", "file_mtime_ns"]).drop_duplicates("id", keep="last")
    successes.to_parquet(out / "recovered_successes.parquet", index=False)
    managed = set(frames["embedded"]["id"]) | set(frames["unembedded"]["id"])
    recovered_ids = set(successes["id"].astype(int))
    new = successes[~successes["id"].isin(managed)][["id", "url", "text"]].copy()

    probe_path = out / "origin_probes.parquet"
    probes = run_network_checks(frames["remaining"], out) if args.network else (
        pd.read_parquet(probe_path) if probe_path.exists() else pd.DataFrame())
    if not probes.empty:
        checked = pd.to_datetime(probes["checked_at"], utc=True)
        age = pd.Timestamp.now(tz="UTC") - checked
        if age.max() > pd.Timedelta(hours=24) or age.min() < -pd.Timedelta(minutes=1):
            raise ValueError("Probe evidence is stale or future-dated")
        mapping = official.set_index("id")["url"]
        if not probes["id"].map(mapping).eq(probes["url"]).all():
            raise ValueError("Probe identity differs from official mapping")
    dead = verified_dead(probes, recovered_ids | managed) & set(frames["remaining"]["id"])
    updated = {name: frame.copy() for name, frame in frames.items()}
    updated["unembedded"] = pd.concat([frames["unembedded"], new], ignore_index=True)
    updated["remaining"] = frames["remaining"][~frames["remaining"]["id"].isin(recovered_ids | dead)].reset_index(drop=True)
    excluded_new = frames["remaining"][frames["remaining"]["id"].isin(dead)].copy()
    excluded_new["reason"] = "final_audit_two_origin_http_404_410"
    excluded_new["excluded_at"] = datetime.now(ZoneInfo("Asia/Bangkok")).isoformat()
    # Recovered text takes precedence even over a previous exclusion.
    updated["excluded"] = pd.concat([
        frames["excluded"][~frames["excluded"]["id"].isin(recovered_ids)], excluded_new
    ], ignore_index=True)
    validate_partition(official, updated)

    latest = history.sort_values("file_mtime_ns").drop_duplicates("id", keep="last")
    matrix = frames["remaining"].merge(
        latest[["id", "status", "reason", "source_file", "ledger_file"]], on="id", how="left", validate="one_to_one")
    matrix["domain"] = matrix["url"].map(domain)
    matrix["decision"] = "keep_retryable_or_unverified"
    matrix.loc[matrix["id"].isin(recovered_ids), "decision"] = "recovered_move_to_unembedded"
    matrix.loc[matrix["id"].isin(dead), "decision"] = "exclude_two_origin_404_410"
    matrix.to_parquet(out / "remaining_decisions.parquet", index=False)
    summary = {
        "checked_at": datetime.now(ZoneInfo("Asia/Bangkok")).isoformat(),
        "mode": "apply" if args.apply else "dry_run",
        "history_files": len(history_parts), "history_rows": len(history),
        "history_unique_ids": int(history["id"].nunique()),
        "history_status_counts": history["status"].value_counts().to_dict(),
        "recovered_unique_successes": len(successes), "new_unembedded": len(new),
        "new_unembedded_ids": [int(value) for value in new["id"]],
        "removed_successes_from_remaining": int(frames["remaining"]["id"].isin(recovered_ids).sum()),
        "excluded_dead_urls": len(dead), "excluded_ids": sorted(dead),
        "origin_probe_requests": len(probes),
        "origin_probe_unique_ids": int(probes["id"].nunique()) if not probes.empty else 0,
        "before_counts": {name: len(frame) for name, frame in frames.items()},
        "after_counts": {name: len(frame) for name, frame in updated.items()},
        "remaining_by_domain": updated["remaining"]["url"].map(domain).value_counts().to_dict(),
        "latest_reasons": matrix["reason"].fillna("no_history").value_counts().to_dict(),
        "input_sha256": before_hashes, "partition_invariant": "PASS",
        "notes": ["Only individual twice-confirmed original-source HTTP 404/410 can be excluded.",
                  "Domain samples do not verify every URL and never authorize bulk exclusion.",
                  "Europe PMC misses are not evidence of deletion on PubMed.",
                  "Historical file mtime orders files, not individual requests."],
    }
    if args.apply:
        summary["changed_registries"] = apply_changes(frames, updated, before_hashes, out)
        disk = {name: pd.read_parquet(path, columns=["id", "url"])
                for name, path in REGISTRIES.items()}
        validate_partition(official, disk)
        summary["output_sha256"] = {name: digest(path) for name, path in REGISTRIES.items()}
    (out / ("applied_summary.json" if args.apply else "dry_run_summary.json")).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    printable = summary.copy()
    if len(printable["new_unembedded_ids"]) > 20:
        printable["new_unembedded_ids"] = {"count": len(summary["new_unembedded_ids"]),
                                          "first_10": summary["new_unembedded_ids"][:10]}
    print(json.dumps(printable, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
