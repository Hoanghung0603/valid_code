"""Build an offline ICD-10 reference snapshot and route all competition queries.

Sources are official WHO ICD-10 2019 plain-text metadata and the Vietnamese
ICD-10 chapter list exposed by the Ministry of Health clinical-coding site.
The generated routes are diagnostic features; they must not be treated as
ground-truth diagnoses or used as hard filters without an official ablation.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.icd10_router import ICD10ChapterRouter, ICDEntry  # noqa: E402


WHO_URL = "https://icdcdn.who.int/icd10/meta/icd102019enMeta.zip"
KCB_ROOT_URL = "https://ccs.whiteneuron.com/api/ICD10_TT06/root"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queries",
        type=Path,
        default=ROOT / "data/raw/btc_provided/query_translated.parquet",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data/reference/icd10",
    )
    parser.add_argument("--timeout", type=float, default=60.0)
    return parser.parse_args()


def download_json(url: str, timeout: float) -> dict:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.json()


def load_who_reference(timeout: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    response = requests.get(WHO_URL, timeout=timeout)
    response.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        chapters_raw = archive.read("icd102019syst_chapters.txt").decode("utf-8-sig")
        codes_raw = archive.read("icd102019syst_codes.txt").decode("utf-8-sig")

    chapter_rows = []
    for line in chapters_raw.splitlines():
        if not line.strip():
            continue
        chapter, name = line.rstrip("\r").split(";", 1)
        chapter_rows.append({"chapter": chapter, "name_en": name})

    entry_rows = []
    for line in codes_raw.splitlines():
        fields = line.rstrip("\r").split(";")
        if len(fields) < 11 or fields[0] not in {"3", "4"}:
            continue
        chapter = fields[3]
        code = fields[5].replace(".-", "")
        descriptions = list(dict.fromkeys(value.strip() for value in fields[8:12] if value.strip()))
        entry_rows.append(
            {
                "chapter": chapter,
                "code": code,
                "level": int(fields[0]),
                "text_en": "; ".join(descriptions),
            }
        )
    return pd.DataFrame(chapter_rows), pd.DataFrame(entry_rows).drop_duplicates(["code", "text_en"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    chapters, entries = load_who_reference(args.timeout)
    kcb = download_json(KCB_ROOT_URL, args.timeout)
    vi_rows = [
        {"chapter": f"{index:02d}", "name_vi": item["data"]["name"]}
        for index, item in enumerate(kcb["data"], start=1)
    ]
    chapters = chapters.merge(pd.DataFrame(vi_rows), on="chapter", how="left")

    router = ICD10ChapterRouter(
        [ICDEntry(row.code, row.chapter, row.text_en) for row in entries.itertuples(index=False)]
    )
    queries = pd.read_parquet(args.queries)
    required = {"id", "query", "query_en"}
    missing = required.difference(queries.columns)
    if missing:
        raise ValueError(f"Missing query columns: {sorted(missing)}")

    route_rows = []
    for row in queries.itertuples(index=False):
        result = router.route(str(row.query), str(row.query_en), top_k=3)
        item = {
            "id": int(row.id),
            "confidence": result.confidence,
            "unknown": result.unknown,
            "margin": result.margin,
            "matched_terms": result.matched_terms,
        }
        for rank in range(3):
            route = result.routes[rank] if rank < len(result.routes) else None
            item[f"chapter_{rank + 1}"] = route.chapter if route else None
            item[f"score_{rank + 1}"] = route.score if route else 0.0
        route_rows.append(item)
    routes = pd.DataFrame(route_rows)

    chapters.to_parquet(args.output_dir / "chapters.parquet", index=False)
    entries.to_parquet(args.output_dir / "entries_en.parquet", index=False)
    routes.to_parquet(args.output_dir / "query_routes.parquet", index=False)

    chapter_names = chapters.set_index("chapter")["name_vi"].to_dict()
    distribution = (
        routes.assign(name_vi=routes["chapter_1"].map(chapter_names))
        .groupby(["chapter_1", "name_vi"], dropna=False)
        .size()
        .sort_values(ascending=False)
    )
    summary = {
        "sources": {"who": WHO_URL, "vietnamese_chapters": KCB_ROOT_URL},
        "queries": int(len(routes)),
        "icd_entries": int(len(entries)),
        "confidence": {key: int(value) for key, value in routes["confidence"].value_counts().items()},
        "unknown": int(routes["unknown"].sum()),
        "top_chapter_distribution": [
            {
                "chapter": None if pd.isna(chapter) else str(chapter),
                "name_vi": None if pd.isna(name) else str(name),
                "queries": int(count),
            }
            for (chapter, name), count in distribution.items()
        ],
        "warning": "Uncalibrated multi-label retrieval prior; never hard-filter low-confidence/UNKNOWN queries.",
    }
    (args.output_dir / "routing_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
