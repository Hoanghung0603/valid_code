"""Download the Vietnamese ICD-10 tree from the Ministry of Health lookup site (icd.kcb.vn).

The site is an Angular app backed by a JSON API (found in its main.js):
    GET https://ccs.whiteneuron.com/api/<VERSION>/root
    GET https://ccs.whiteneuron.com/api/<VERSION>/childs/<model>?id=<node id>
    GET https://ccs.whiteneuron.com/api/<VERSION>/data/<model>?id=<node id>   (leaf details)
VERSION: ICD10 (QĐ 4469/QĐ-BYT, 2020) or ICD10_TT06 (Thông tư 06/2026/TT-BYT).
Tree: chapter -> section -> type (3-char code) -> disease (4-char code, leaf).

One request at a time with a pause (polite). Resumable: finished nodes are cached in
<out>/cache_<version>.jsonl.

Usage:
    python -I terms/icd10_vn.py --version ICD10 --out D:/r2ai_data/external/icd10 [--details]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

API = "https://ccs.whiteneuron.com/api/"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                         "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
           "Referer": "https://icd.kcb.vn/", "Origin": "https://icd.kcb.vn"}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="ICD10", choices=["ICD10", "ICD10_TT06"])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pause", type=float, default=0.4)
    ap.add_argument("--details", action="store_true", help="also fetch leaf details (notes, include/exclude)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / f"cache_{args.version}.jsonl"
    cache: dict[str, dict] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            rec = json.loads(line)
            cache[rec["key"]] = rec["resp"]
    s = requests.Session()
    s.headers.update(HEADERS)
    cache_f = open(cache_path, "a", encoding="utf-8", newline="\n")

    def call(path: str, node_id: str | None = None) -> dict:
        key = f"{path}?{node_id}"
        if key in cache:
            return cache[key]
        for attempt in range(4):
            try:
                r = s.get(API + args.version + "/" + path,
                          params={"id": node_id} if node_id else None, timeout=30)
                if r.status_code == 200:
                    resp = r.json()
                    break
            except (requests.RequestException, ValueError):
                pass
            time.sleep(5 * (attempt + 1))
        else:
            raise RuntimeError(f"failed: {key}")
        time.sleep(args.pause)
        cache[key] = resp
        cache_f.write(json.dumps({"key": key, "resp": resp}, ensure_ascii=False) + "\n")
        cache_f.flush()
        return resp

    rows = []

    def walk(item: dict, parents: dict) -> None:
        d = item["data"]
        row = {"model": item["model"], "id": item["id"], "code": d.get("code"),
               "name_vi": d.get("name"), "parent_id": parents.get("_last"),
               **{f"{k}_id": v for k, v in parents.items() if not k.startswith("_")}}
        if args.details and item.get("is_leaf"):
            det = call(f"data/{item['model']}", item["id"]).get("data", {}).get("data", {})
            row.update(note=det.get("note"), include=det.get("include"), exclude=det.get("exclude"))
        rows.append(row)
        if len(rows) % 500 == 0:
            print(f"  {len(rows)} nodes", flush=True)
        if not item.get("is_leaf"):
            p = {**parents, item["model"]: item["id"], "_last": item["id"]}
            for child in call(f"childs/{item['model']}", item["id"]).get("data") or []:
                walk(child, p)

    for chapter in call("root").get("data") or []:
        walk(chapter, {})
    cache_f.close()
    df = pd.DataFrame(rows)
    out = args.out / f"icd10_vi_{args.version}.parquet"
    df.to_parquet(out, index=False)
    print(out, len(df), "nodes;", df.model.value_counts().to_dict())


if __name__ == "__main__":
    main()
