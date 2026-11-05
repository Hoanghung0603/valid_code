"""Offline inventory and approval-only recovery proposals; never fetch or change state.

Example: python deploy/crawl_recovery_plan.py --corpus data/links_corpus.parquet
  --state data/crawl/state --out data/crawl/recovery --tiers 1,2,3

Writes inventory.json and candidates.jsonl exclusively (existing files are refused).
--out must be inside the crawl directory containing --state, outside state itself.
Missing summaries refer to verified state-ok records absent from shards, NOT gaps
in the corpus ID range. A complete, unique missing_samples list is required.
"""
import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


# Deliberately copied from scripts/crawl.py; parity is checked by tests. Importing
# the crawler would also load extraction/network dependencies and mutate sys.path.
MAX_ATTEMPTS = 3
TIER2_HOSTS = {
    "120ask.com", "ask.39.net", "jbk.39.net", "a-hospital.com", "familydoctor.com.cn", "cnkang.com",
    "youlai.cn", "msdmanuals.cn", "baidu.com", "ask.familydoctor.com.cn", "jb39.com",
}
HARD_HOSTS = {"zysjonline.com", "nhathuoclongchau.com.vn", "tamanhhospital.vn", "vov.vn",
              "pmc-ecm-healthblog.beta.pharmacity.io", "article.iiyi.com", "bingli.iiyi.com"}
INSPECTION_HOSTS = {
    "zydcd.com": "inspect_short_content",
    "zhongyibaodian.net": "inspect_short_content",
    "a-hospital.com": "inspect_short_content",
    "msdmanuals.cn": "inspect_extraction",
    "laodong.vn": "inspect_extraction_requires_specific_approval",
    "baotayninh.vn": "inspect_extraction_requires_specific_approval",
}
OUTPUT_NAMES = ("inventory.json", "candidates.jsonl")


def host_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def tier_of(host: str) -> int:
    if host in HARD_HOSTS:
        return 4
    if host.endswith(".vn") or host in {"hellobacsi.com", "vinmec.com", "suckhoeviet.org.vn", "benhvienvietduc.org"}:
        return 1
    if host in TIER2_HOSTS:
        return 2
    return 3


@dataclass
class History:
    last_outcome: str = None
    nonterminal_attempts: int = 0
    total_attempts: int = 0
    ever_ok: bool = False
    ever_permanent: bool = False
    ever_blocked: bool = False
    transient_only: bool = True
    ever_redirect: bool = False
    outcomes: Counter = field(default_factory=Counter)

    @property
    def terminal_state(self):
        # Any terminal history makes State.pending false, even after later errors.
        if self.ever_ok:
            return "ok"
        return "permanent" if self.ever_permanent else None

    def record(self, outcome):
        self.last_outcome = outcome
        self.total_attempts += 1
        self.outcomes[outcome] += 1
        self.ever_ok |= outcome == "ok"
        self.ever_permanent |= outcome == "permanent"
        if outcome not in ("ok", "permanent"):
            self.nonterminal_attempts += 1
            self.transient_only &= outcome == "transient" or outcome.startswith("transient:")
        self.ever_blocked |= outcome == "blocked" or outcome.startswith("blocked:")
        self.ever_redirect |= outcome == "transient:TooManyRedirects"


def read_state(state: Path):
    """Read sorted logs only. Unlike State, reject damaged/blank/unfinished lines.

    Nonblank raw outcomes (including ok:error/permanent:error) remain
    nonterminal unless they are the EXACT literals ok or permanent.
    """
    histories = {}
    diagnostics = Counter(files=0, lines=0, valid_lines=0, malformed_lines=0,
                          empty_lines=0, blank_outcomes=0, truncated_lines=0)
    for path in sorted(state.glob("log-*.tsv")):
        diagnostics["files"] += 1
        with path.open("rb") as fh:
            for raw in fh:
                diagnostics["lines"] += 1
                if not raw.endswith(b"\n"):
                    diagnostics["truncated_lines"] += 1
                    continue
                try:
                    line = raw.decode("utf-8").removesuffix("\n").removesuffix("\r")
                    if not line:
                        diagnostics["empty_lines"] += 1
                        continue
                    sid, outcome = line.split("\t")
                    sid = int(sid)
                    if not outcome.strip():
                        diagnostics["blank_outcomes"] += 1
                        continue
                    if any(ord(c) < 32 for c in outcome):
                        raise ValueError("control character in outcome")
                except (ValueError, UnicodeError):
                    diagnostics["malformed_lines"] += 1
                    continue
                histories.setdefault(sid, History()).record(outcome)
                diagnostics["valid_lines"] += 1
    return histories, dict(diagnostics)


def read_missing_summary(path):
    if path is None:
        return set()
    summary = json.loads(path.read_text(encoding="utf-8"))
    count, samples = summary.get("missing"), summary.get("missing_samples")
    if type(count) is not int or count < 0 or not isinstance(samples, list):
        raise ValueError("missing summary requires missing count and complete missing_samples list")
    if any(type(i) is not int for i in samples):
        raise ValueError("missing_samples must contain integer IDs")
    if len(samples) != count or len(set(samples)) != count:
        raise ValueError("missing_samples is incomplete or duplicated; refuse partial export")
    return set(samples)


def decision(host, history, blocked_hosts, missing=False):
    if host in HARD_HOSTS or host in blocked_hosts:
        return False, "archive_host_blocked"
    if missing:
        return True, "recover_missing_shard_record"
    if history.ever_ok:
        return False, "archive_already_ok"
    if not history.total_attempts:
        return False, "archive_never_attempted"
    if history.ever_permanent:
        if host in INSPECTION_HOSTS:
            return True, INSPECTION_HOSTS[host]
        if host == "qdnd.vn" and history.ever_redirect:
            return True, "inspect_redirects"
        return False, "archive_permanent"
    if host == "qdnd.vn" and history.ever_redirect:
        return True, "inspect_redirects"
    if history.transient_only and history.nonterminal_attempts < MAX_ATTEMPTS:
        return True, "propose_transient_retry"
    if history.nonterminal_attempts >= MAX_ATTEMPTS:
        return False, "archive_retry_limit"
    return False, "archive_other_outcome"


def csv_set(value):
    return {item.strip() for item in value.split(",") if item.strip()} if value else set()


def check_output_paths(corpus, state, out, missing_summary):
    crawl = state.parent
    if out == crawl or crawl not in out.parents:
        raise ValueError("--out must be a subdirectory within the crawl directory containing --state")
    if out == state or state in out.parents:
        raise ValueError("--out must be outside the input state directory")
    for source in (corpus, missing_summary):
        if source is not None and (out == source or out in source.parents):
            raise ValueError("--out cannot contain an input file")
    for name in OUTPUT_NAMES:
        destination = out / name
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"Refuse existing output: {destination}")


def generate_plan(corpus, state, out, hosts=None, exclude_hosts=None, tiers=None, missing_summary=None):
    corpus, state, out = (Path(p).resolve() for p in (corpus, state, out))
    missing_summary = Path(missing_summary).resolve() if missing_summary is not None else None
    if not corpus.is_file() or not state.is_dir():
        raise ValueError("--corpus must exist and --state must be an existing directory")
    check_output_paths(corpus, state, out, missing_summary)
    hosts, exclude_hosts = csv_set(hosts), csv_set(exclude_hosts)
    tiers = {int(t) for t in csv_set(tiers)}
    if tiers - {1, 2, 3, 4}:
        raise ValueError("--tiers must contain only 1,2,3,4")
    missing = read_missing_summary(missing_summary)
    df = pd.read_parquet(corpus, columns=["id", "url"])
    if df["id"].isna().any() or df["id"].duplicated().any():
        raise ValueError("corpus IDs must be non-null and unique")
    if not pd.api.types.is_integer_dtype(df["id"]):
        raise ValueError("corpus IDs must be integers")
    if not df["url"].map(lambda u: isinstance(u, str)).all():
        raise ValueError("corpus URLs must be strings")
    df["host"] = df["url"].map(host_of)
    df["tier"] = df["host"].map(tier_of)
    id_hosts = dict(zip(df["id"], df["host"]))
    unknown_missing = missing - id_hosts.keys()
    if unknown_missing:
        raise ValueError(f"missing_samples IDs absent from corpus: {sorted(unknown_missing)}")
    histories, diagnostics = read_state(state)
    blocked_hosts = {id_hosts[i] for i, h in histories.items() if h.ever_blocked and i in id_hosts}
    diagnostics["ids_absent_from_corpus"] = len(histories.keys() - id_hosts.keys())
    diagnostics["attempts_absent_from_corpus"] = sum(
        h.total_attempts for i, h in histories.items() if i not in id_hosts)
    corpus_rows = len(df)
    if hosts:
        df = df[df["host"].isin(hosts)]
    if exclude_hosts:
        df = df[~df["host"].isin(exclude_hosts)]
    if tiers:
        df = df[df["tier"].isin(tiers)]
    totals, by_host, candidates = Counter(), {}, []
    scope_missing = set()
    for sid, url, host, tier in df.itertuples(index=False, name=None):
        sid = int(sid)
        h = histories.get(sid, History())
        is_missing = sid in missing
        if is_missing:
            scope_missing.add(sid)
        eligible, reason = decision(host, h, blocked_hosts, is_missing)
        bucket = by_host.setdefault(host, {"tier": tier, "ever_blocked": host in blocked_hosts,
                                           "hard_host": host in HARD_HOSTS, "counts": Counter(),
                                           "reasons": Counter(), "raw_outcomes": Counter()})
        metrics = {"urls": 1, "attempted": int(h.total_attempts > 0),
                   "never_attempted": int(h.total_attempts == 0),
                   "ever_ok": int(h.ever_ok), "ever_permanent": int(h.ever_permanent),
                   "terminal": int(h.terminal_state is not None),
                   "total_attempts": h.total_attempts, "nonterminal_attempts": h.nonterminal_attempts,
                   "missing_shard_records": int(is_missing),
                   "candidates": int(eligible), "archived": int(not eligible)}
        totals.update(metrics)
        bucket["counts"].update(metrics)
        bucket["reasons"][reason] += 1
        bucket["raw_outcomes"].update(h.outcomes)
        if eligible:
            candidates.append({"id": sid, "url": url, "host": host, "last_outcome": h.last_outcome,
                               "nonterminal_attempts": h.nonterminal_attempts,
                               "total_attempts": h.total_attempts, "reason": reason,
                               "requires_approval": True, "terminal_state": h.terminal_state})
    report = {
        "schema_version": 1, "offline": True, "requires_approval_before_network_or_restart": True,
        "inputs": {"corpus": str(corpus), "state": str(state),
                   "missing_summary": str(missing_summary) if missing_summary else None},
        "filters": {"hosts": sorted(hosts), "exclude_hosts": sorted(exclude_hosts), "tiers": sorted(tiers)},
        "corpus_rows": corpus_rows, "state_diagnostics": diagnostics,
        "blocked_hosts_all_corpus": sorted(blocked_hosts),
        "missing_summary": {"count": len(missing), "in_scope": len(scope_missing),
                            "excluded_by_filters": len(missing - scope_missing)},
        "totals": dict(totals), "hosts": dict(sorted(by_host.items())),
        "limitations": ["State does not distinguish short text from HTTP 404/410 or prove JavaScript is required.",
                        "Shard presence is taken from the supplied complete verification summary, not checked here.",
                        "Malformed, blank and non-newline-terminated state lines are counted and ignored.",
                        "Inspection labels are proposals, not diagnoses; no candidate authorizes network access."],
    }
    # Validate everything before creating outputs. Exclusive opens protect existing
    # named outputs even if another process creates one after the initial checks.
    out.mkdir(parents=True, exist_ok=True)
    with (out / OUTPUT_NAMES[0]).open("x", encoding="utf-8") as report_fh:
        with (out / OUTPUT_NAMES[1]).open("x", encoding="utf-8") as candidates_fh:
            json.dump(report, report_fh, ensure_ascii=False, indent=2)
            report_fh.write("\n")
            for candidate in sorted(candidates, key=lambda c: c["id"]):
                candidates_fh.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--hosts", help="Exact normalized hosts, comma separated; no substring matching")
    parser.add_argument("--exclude-hosts")
    parser.add_argument("--tiers")
    parser.add_argument("--missing-summary", type=Path)
    args = parser.parse_args(argv)
    try:
        report = generate_plan(args.corpus, args.state, args.out, args.hosts,
                               args.exclude_hosts, args.tiers, args.missing_summary)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"error: {exc}\n")
    print(f"Inventory: {args.out / OUTPUT_NAMES[0]}; candidates: {report['totals'].get('candidates', 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
