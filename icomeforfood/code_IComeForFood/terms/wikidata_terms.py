"""Trilingual (vi / en / zh) medical terms from Wikidata (CC0).

A "medical term" = a Wikidata item that has one of these identifiers:
    P486  MeSH descriptor ID   (diseases, drugs, anatomy, procedures, tests ...)
    P494  ICD-10
    P267  ATC code             (drugs)
For each item we take its label and aliases in vi, en and zh (zh, zh-hans, zh-cn).
One SPARQL query per (property, language) keeps each query small enough for the
public endpoint.

Output: <out>/wikidata_terms.parquet with columns
    item, prop, code, lang, term, is_alias

Usage:
    python -I terms/wikidata_terms.py --out D:/r2ai_data/external/terms
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "r2ai-vibiomir-research/0.1 (student competition team; github.com/namgau)",
           "Accept": "application/sparql-results+json"}
PROPS = {"P486": "mesh", "P494": "icd10", "P267": "atc"}
LANGS = {"vi": ["vi"], "en": ["en"], "zh": ["zh", "zh-hans", "zh-cn"]}
QUERY = """
SELECT ?item ?code ?term ?isAlias WHERE {{
  ?item wdt:{prop} ?code .
  {{ ?item rdfs:label ?term . BIND(false AS ?isAlias) }}
  UNION
  {{ ?item skos:altLabel ?term . BIND(true AS ?isAlias) }}
  FILTER(LANG(?term) IN ({langs}))
  {bucket}
}}"""
# P486 has ~39K items: split by the last digit of the code so each query finishes
# inside the endpoint's 60 s limit (a timed-out query returns truncated JSON).
BUCKETS = {"P486": [f'FILTER(STRENDS(STR(?code), "{d}"))' for d in "0123456789"]}


def run(query: str) -> list[dict]:
    for attempt in range(4):
        r = requests.get(ENDPOINT, params={"query": query}, headers=HEADERS, timeout=300)
        if r.status_code == 200:
            try:
                # some labels contain raw control characters; strict=False accepts them
                return json.loads(r.text, strict=False)["results"]["bindings"]
            except ValueError as e:  # truncated answer when the server hits its time limit
                print(f"  bad JSON ({e}); retrying", flush=True)
        time.sleep(20 * (attempt + 1))  # 429/503/truncated: back off
    raise RuntimeError("Wikidata query failed 4 times")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for prop, pname in PROPS.items():
        for lang, tags in LANGS.items():
            n, t0 = 0, time.time()
            for bucket in BUCKETS.get(prop, [""]):
                q = QUERY.format(prop=prop, langs=", ".join(f'"{t}"' for t in tags), bucket=bucket)
                res = run(q)
                n += len(res)
                for b in res:
                    rows.append({"item": b["item"]["value"].rsplit("/", 1)[-1], "prop": pname,
                                 "code": b["code"]["value"], "lang": lang, "term": b["term"]["value"],
                                 "is_alias": b["isAlias"]["value"] == "true"})
                time.sleep(2)
            print(f"{pname:6s} {lang}: {n:7d} rows  {time.time() - t0:5.1f}s", flush=True)
    df = pd.DataFrame(rows).drop_duplicates()
    df.to_parquet(args.out / "wikidata_terms.parquet", index=False)
    print("items per prop and lang (distinct):")
    print(df.groupby(["prop", "lang"]).item.nunique().unstack().to_string())
    print("terms per lang:", df.groupby("lang").term.nunique().to_dict())


if __name__ == "__main__":
    main()
