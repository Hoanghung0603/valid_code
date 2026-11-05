"""Audit QuerySpec/atom artifacts before retrieval or production expansion."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/audits/queryspec_pilot_v6_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--min-usable-rate", type=float, default=0.95)
    parser.add_argument("--expected-model", default="qwen/qwen3-14b")
    parser.add_argument("--expected-provider", default="DeepInfra")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs_path = args.input_dir / "query_specs.parquet"
    atoms_path = args.input_dir / "evidence_atoms.parquet"
    specs = pd.read_parquet(specs_path)
    atoms = pd.read_parquet(atoms_path)

    failures: list[str] = []
    if specs["query_id"].duplicated().any():
        failures.append("duplicate query_id in query_specs.parquet")
    if atoms.duplicated(["query_id", "atom_id"]).any():
        failures.append("duplicate query_id/atom_id in evidence_atoms.parquet")
    usable = specs["status"].eq("ok")
    usable_rate = float(usable.mean()) if len(specs) else 0.0
    if usable_rate < args.min_usable_rate:
        failures.append(f"usable rate {usable_rate:.4f} < {args.min_usable_rate:.4f}")
    bad_models = specs.loc[usable & specs["model"].ne(args.expected_model), "query_id"].astype(int).tolist()
    bad_providers = specs.loc[
        usable & specs["provider"].str.casefold().ne(args.expected_provider.casefold()), "query_id"
    ].astype(int).tolist()
    if bad_models:
        failures.append(f"model mismatch for query IDs: {bad_models}")
    if bad_providers:
        failures.append(f"provider mismatch for query IDs: {bad_providers}")

    empty_query_columns: dict[str, list[int]] = {}
    for column in ("retrieval_query_vi", "retrieval_query_en", "retrieval_query_zh"):
        bad = atoms.loc[atoms[column].fillna("").str.strip().eq(""), "query_id"].astype(int).tolist()
        if bad:
            empty_query_columns[column] = bad
            failures.append(f"empty {column} for {len(bad)} atoms")

    expected_atom_counts = specs.loc[usable, ["query_id", "atom_count"]].set_index("query_id")["atom_count"]
    actual_atom_counts = atoms.groupby("query_id").size()
    atom_mismatches = [
        int(query_id)
        for query_id, expected in expected_atom_counts.items()
        if int(actual_atom_counts.get(query_id, 0)) != int(expected)
    ]
    if atom_mismatches:
        failures.append(f"atom count mismatch for query IDs: {atom_mismatches}")

    atom_distribution = Counter(map(int, specs.loc[usable, "atom_count"]))
    report = {
        "input_dir": str(args.input_dir),
        "query_rows": int(len(specs)),
        "unique_query_ids": int(specs["query_id"].nunique()),
        "status_counts": {str(key): int(value) for key, value in specs["status"].value_counts().items()},
        "usable_rate": usable_rate,
        "fallback_query_ids": specs.loc[~usable, "query_id"].astype(int).tolist(),
        "atom_rows": int(len(atoms)),
        "atom_distribution": {str(key): int(value) for key, value in sorted(atom_distribution.items())},
        "mean_atoms_per_usable_query": float(atoms.groupby("query_id").size().mean()),
        "empty_query_columns": empty_query_columns,
        "bad_model_query_ids": bad_models,
        "bad_provider_query_ids": bad_providers,
        "failures": failures,
        "passed": not failures,
    }
    output_path = args.input_dir / "audit.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
