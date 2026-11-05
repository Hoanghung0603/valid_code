"""Compare full ColBERT shortlist rankings with the submitted Exp143 doc sets.

No gold labels are available; novelty and score margins are not relevance metrics.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pandas as pd

from scripts.build_final_sprint_exp142_missing_parents import read_rows


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/kaggle_inputs/colbert_full_20261103/colbert_full_pairs.parquet"
OUTPUT = ROOT / "data/kaggle_outputs/colbert_full_20261104/colbert_full_outputs.zip"
CONTROL = ROOT / "submission_private01_exp143_replay.zip"
QUERY = ROOT / "data/raw/btc_provided/query.parquet"


def main() -> None:
    with zipfile.ZipFile(OUTPUT) as archive:
        scores = pd.read_parquet(io.BytesIO(archive.read("scores.parquet")))
    pairs = pd.read_parquet(INPUT, columns=["pair_id", "query_id", "doc_id", "passage_text"])
    joined = scores.merge(pairs, on=["pair_id", "query_id", "doc_id"], validate="one_to_one")
    # Take the strongest passage per document for this exploratory rerank.
    docs = (joined.sort_values(["query_id", "colbert_score", "doc_id"], ascending=[True, False, True])
            .drop_duplicates(["query_id", "doc_id"]))
    champion = {}
    for row in read_rows(CONTROL):
        qid = int(row["id"])
        if qid in champion:
            raise ValueError("Duplicate query in champion")
        champion[qid] = set(map(int, row["relevant_docs"]))
    if len(champion) != 1200:
        raise ValueError("Champion does not cover all queries")
    query_frame = pd.read_parquet(QUERY, columns=["id", "query"])
    qtext = dict(zip(query_frame["id"].astype(int), query_frame["query"].astype(str)))
    records = []
    examples = []
    for qid, group in docs.groupby("query_id", sort=True):
        qid = int(qid)
        group = group.copy()
        group["is_champion"] = group.doc_id.isin(champion[qid])
        top5 = group.head(5)
        top10 = group.head(10)
        novel = group[~group.is_champion]
        known = group[group.is_champion]
        records.append({
            "query_id": qid,
            "champion_docs": len(champion[qid]),
            "candidate_docs": len(group),
            "novel_in_top5": int((~top5.is_champion).sum()),
            "novel_in_top10": int((~top10.is_champion).sum()),
            "novel_in_50": len(novel),
            "best_novel_score": float(novel.colbert_score.max()) if len(novel) else None,
            "best_known_score": float(known.colbert_score.max()) if len(known) else None,
            "novel_above_known": bool(len(novel) and len(known) and
                                      novel.colbert_score.max() > known.colbert_score.max()),
        })
        if len(novel):
            best = novel.iloc[0]
            examples.append({
                "query_id": qid,
                "query": qtext[qid],
                "doc_id": int(best.doc_id),
                "score": round(float(best.colbert_score), 4),
                "best_known_score": round(float(known.colbert_score.max()), 4) if len(known) else None,
                "passage_preview": str(best.passage_text)[:500],
            })
    frame = pd.DataFrame(records)
    examples.sort(key=lambda x: x["score"] - (x["best_known_score"] or 0), reverse=True)
    report = {
        "disclaimer": "Ranking/novelty only; no gold labels or private score",
        "queries": len(frame),
        "queries_with_novel_top5": int(frame.novel_in_top5.gt(0).sum()),
        "queries_with_novel_top10": int(frame.novel_in_top10.gt(0).sum()),
        "mean_novel_top5": round(float(frame.novel_in_top5.mean()), 3),
        "mean_novel_top10": round(float(frame.novel_in_top10.mean()), 3),
        "mean_novel_50": round(float(frame.novel_in_50.mean()), 3),
        "queries_best_novel_above_best_known": int(frame.novel_above_known.sum()),
        "samples_largest_novel_score_margin": examples[:12],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
