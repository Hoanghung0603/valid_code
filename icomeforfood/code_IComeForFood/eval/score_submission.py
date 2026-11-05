"""Offline replica of the official R2AI ViBioMIR scorer (doc + chunk P/R/F2, macro).

Rules implemented (from the competition page):
- Doc level:   P = |D∩G|/|D|, R = |D∩G|/|G|.
- Chunk level: a submitted chunk is correct if overlap(c, g) = LCS_tokens(c, g)/|g| >= 0.4
               for at least one reference chunk g of the same query and same doc_id.
               P = correct submitted chunks / submitted chunks
               R = reference chunks matched by >= 1 submitted chunk / reference chunks
- Text is normalized (NFKC, HTML unescape, lowercase, collapse whitespace, strip) first.
- F2 per query, then macro-average over queries.

Assumptions (not stated by the organizers, confirm when possible):
- Tokens = whitespace split after normalization (--tokenizer ws). Use --tokenizer cjk
  to see how much Chinese scores would change if the scorer splits CJK characters.
- Empty prediction or empty gold at a level -> P = R = F2 = 0 for that query/level.

Gold format (our dev sets), JSON list:
    [{"id": int, "relevant_docs": [int], "relevant_chunks": [{"doc_id": int, "chunk_text": str}]}]

Usage:
    python eval/score_submission.py --pred submission.json --gold dev_gold.json [--tokenizer ws]
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r2ai_eval import f_beta, overlap, tokenize  # noqa: E402

THRESHOLD = 0.4


def prf(n_correct_p: int, n_pred: int, n_found_r: int, n_gold: int) -> tuple[float, float, float]:
    p = n_correct_p / n_pred if n_pred else 0.0
    r = n_found_r / n_gold if n_gold else 0.0
    return p, r, f_beta(p, r)


def score_query(pred: dict, gold: dict, tok: str) -> dict:
    d_pred = {int(x) for x in pred.get("relevant_docs", [])}
    d_gold = {int(x) for x in gold.get("relevant_docs", [])}
    inter = len(d_pred & d_gold)
    dp, dr, df = prf(inter, len(d_pred), inter, len(d_gold))

    gold_by_doc: dict[int, list[list[str]]] = {}
    for g in gold.get("relevant_chunks", []):
        gold_by_doc.setdefault(int(g["doc_id"]), []).append(tokenize(g["chunk_text"], tok))
    found = {doc: [False] * len(gs) for doc, gs in gold_by_doc.items()}

    correct = 0
    preds = pred.get("relevant_chunks", [])
    for c in preds:
        doc = int(c["doc_id"])
        refs = gold_by_doc.get(doc)
        if not refs:
            continue
        ct = tokenize(c["chunk_text"], tok)
        hit = False
        for i, g in enumerate(refs):
            if overlap(ct, g) >= THRESHOLD:
                hit = True
                found[doc][i] = True
        correct += hit
    n_gold_chunks = sum(len(v) for v in gold_by_doc.values())
    n_found = sum(sum(v) for v in found.values())
    cp, cr, cf = prf(correct, len(preds), n_found, n_gold_chunks)
    return {"doc_p": dp, "doc_r": dr, "doc_f2": df, "chunk_p": cp, "chunk_r": cr, "chunk_f2": cf}


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--tokenizer", default="ws", choices=["ws", "cjk"])
    ap.add_argument("--per-query", type=Path, help="optional CSV with per-query scores")
    args = ap.parse_args()

    preds = {int(q["id"]): q for q in json.loads(args.pred.read_text(encoding="utf-8"))}
    golds = {int(q["id"]): q for q in json.loads(args.gold.read_text(encoding="utf-8"))}
    missing = sorted(set(golds) - set(preds))
    if missing:
        print(f"WARNING: {len(missing)} gold queries missing from prediction "
              f"(the real leaderboard rejects such files), e.g. {missing[:5]}")

    rows = []
    for qid, gold in sorted(golds.items()):
        s = score_query(preds.get(qid, {}), gold, args.tokenizer)
        rows.append({"id": qid, **s})

    keys = ["doc_p", "doc_r", "doc_f2", "chunk_p", "chunk_r", "chunk_f2"]
    macro = {k: statistics.fmean(r[k] for r in rows) for k in keys}
    print(f"queries scored: {len(rows)}  tokenizer: {args.tokenizer}")
    print(f"DOC   P={macro['doc_p']:.4f}  R={macro['doc_r']:.4f}  F2={macro['doc_f2']:.4f}")
    print(f"CHUNK P={macro['chunk_p']:.4f}  R={macro['chunk_r']:.4f}  F2={macro['chunk_f2']:.4f}")

    if args.per_query:
        import csv
        with open(args.per_query, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["id", *keys])
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
