"""Offline re-implementation of the R2AI2026 scoring (doc level + chunk level).

Metrics (computed per query, then macro-averaged over the scored queries):

* ``P  = hits / |pred|``            (0 if ``|pred| == 0``)
* ``R  = hits / |gold|``
* ``F2 = 5PR / (4P + R)``           (0 if ``P == R == 0``)
* ``MRR@5`` = 1/rank of the first correct prediction among the top-5 (0 if none)

Doc level: predictions and gold are *sets* of doc_id strings (duplicates in the
prediction are dropped, first occurrence keeps its rank).

Chunk level: a predicted chunk can only match a gold chunk with the same
``doc_id``. Matching modes (``chunk_match``):

* ``exact``   – equal after NFC + whitespace collapsing (case sensitive).
* ``contain`` – one normalised string contains the other.
* ``overlap`` – token overlap ``>= overlap_threshold``; tokens are lower-cased
  words for VI/EN and single characters for CJK. ``overlap_metric`` is ``f1``
  (default, multiset token F1), ``recall`` (fraction of gold tokens covered) or
  ``jaccard``.

Hits are a maximum one-to-one bipartite matching between predicted and gold
chunks, so a gold chunk is never counted twice and two near-duplicate
predictions of one gold chunk give one hit (the other is a false positive).
Exact duplicate predicted chunks (same doc_id + normalised text) are dropped
before matching, mirroring the set semantics used at doc level.

Edge cases:

* Query in gold but absent from the prediction -> empty prediction (all 0).
* Query in prediction but not in gold -> ignored (counted in ``n_extra_pred``).
* Gold empty at a level (``empty_gold``):
  - ``"skip"`` (default): the query is excluded from that level's macro average
    (useful for dev sets that only label docs, or have unlabeled queries).
  - ``"strict"``: score 1 for every metric if the prediction is also empty,
    otherwise 0.

CLI::

    python -m src.evaluate --pred sub.zip --gold gold.json [--chunk-match contain]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
import zipfile
from functools import lru_cache
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CHUNK_MATCH_MODES = ("exact", "contain", "overlap")

_WS = re.compile(r"\s+")
_ZW = re.compile(r"[​‌‍﻿]")
_CJK = "㐀-䶿一-鿿豈-﫿぀-ヿ가-힯"
_TOKEN = re.compile(rf"[{_CJK}]|[^\W_{_CJK}]+", re.UNICODE)


# --------------------------------------------------------------------------- io
def load_json_any(obj: Any) -> Any:
    """Load a submission/gold from a list (returned as is), .json, .jsonl or .zip path."""
    if not isinstance(obj, (str, Path)):
        return obj
    p = Path(obj)
    if p.suffix == ".zip":
        with zipfile.ZipFile(p) as zf:
            names = [n for n in zf.namelist() if n.endswith(".json") and not n.endswith("/")]
            if len(names) != 1:
                raise ValueError(f"{p}: expected exactly one .json in zip, found {names}")
            return json.loads(zf.read(names[0]).decode("utf-8"))
    text = p.read_text(encoding="utf-8")
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    return json.loads(text)


def _by_id(entries: Iterable[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for e in entries or []:
        try:
            qid = int(e["id"])
        except (KeyError, TypeError, ValueError):
            continue
        out[qid] = e
    return out


# ------------------------------------------------------------------ text utils
@lru_cache(maxsize=200_000)
def norm_text(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "")
    s = _ZW.sub("", s)
    return _WS.sub(" ", s).strip()


@lru_cache(maxsize=200_000)
def tokens(s: str) -> tuple[str, ...]:
    return tuple(_TOKEN.findall(norm_text(s).lower()))


def token_overlap(pred: str, gold: str, metric: str = "f1") -> float:
    tp, tg = tokens(pred), tokens(gold)
    if not tp or not tg:
        return 0.0
    if metric == "jaccard":
        sp, sg = set(tp), set(tg)
        return len(sp & sg) / len(sp | sg)
    common = sum((Counter(tp) & Counter(tg)).values())
    if common == 0:
        return 0.0
    if metric == "recall":
        return common / len(tg)
    if metric == "f1":
        p, r = common / len(tp), common / len(tg)
        return 2 * p * r / (p + r)
    raise ValueError(f"unknown overlap_metric {metric!r}")


def chunk_matches(pred: dict, gold: dict, mode: str = "exact",
                  threshold: float = 0.5, metric: str = "f1") -> bool:
    if str(pred.get("doc_id")) != str(gold.get("doc_id")):
        return False
    a, b = norm_text(pred.get("chunk_text") or ""), norm_text(gold.get("chunk_text") or "")
    if not a or not b:
        return False
    if mode == "exact":
        return a == b
    if mode == "contain":
        return a in b or b in a
    if mode == "overlap":
        return token_overlap(a, b, metric) >= threshold
    raise ValueError(f"unknown chunk_match {mode!r}; expected one of {CHUNK_MATCH_MODES}")


def max_bipartite_matching(adj: list[list[int]], n_right: int) -> int:
    """Kuhn's algorithm. adj[i] = right nodes compatible with left node i."""
    match_r = [-1] * n_right

    def try_assign(i: int, seen: list[bool]) -> bool:
        for j in adj[i]:
            if not seen[j]:
                seen[j] = True
                if match_r[j] == -1 or try_assign(match_r[j], seen):
                    match_r[j] = i
                    return True
        return False

    return sum(try_assign(i, [False] * n_right) for i in range(len(adj)))


# --------------------------------------------------------------- per-query
def prf2(hits: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = hits / n_pred if n_pred else 0.0
    r = hits / n_gold if n_gold else 0.0
    f2 = 5 * p * r / (4 * p + r) if (p + r) > 0 else 0.0
    return p, r, f2


def _dedupe_docs(docs: Iterable[Any]) -> list[str]:
    seen, out = set(), []
    for d in docs or []:
        d = str(d)
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def _dedupe_chunks(chunks: Iterable[dict]) -> list[dict]:
    seen, out = set(), []
    for c in chunks or []:
        if not isinstance(c, dict):
            continue
        key = (str(c.get("doc_id")), norm_text(c.get("chunk_text") or ""))
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _empty_gold_scores(n_pred: int) -> dict:
    v = 1.0 if n_pred == 0 else 0.0
    return {"P": v, "R": v, "F2": v, "MRR@5": v, "hits": 0, "n_pred": n_pred, "n_gold": 0}


def score_docs(pred_docs, gold_docs, k_mrr: int = 5) -> dict:
    pred, gold = _dedupe_docs(pred_docs), set(_dedupe_docs(gold_docs))
    hits = sum(1 for d in pred if d in gold)
    p, r, f2 = prf2(hits, len(pred), len(gold))
    mrr = next((1.0 / (i + 1) for i, d in enumerate(pred[:k_mrr]) if d in gold), 0.0)
    return {"P": p, "R": r, "F2": f2, "MRR@5": mrr, "hits": hits,
            "n_pred": len(pred), "n_gold": len(gold)}


def score_chunks(pred_chunks, gold_chunks, mode: str = "exact", threshold: float = 0.5,
                 metric: str = "f1", k_mrr: int = 5) -> dict:
    pred, gold = _dedupe_chunks(pred_chunks), _dedupe_chunks(gold_chunks)
    adj = [[j for j, g in enumerate(gold) if chunk_matches(c, g, mode, threshold, metric)]
           for c in pred]
    hits = max_bipartite_matching(adj, len(gold)) if gold else 0
    p, r, f2 = prf2(hits, len(pred), len(gold))
    mrr = next((1.0 / (i + 1) for i, a in enumerate(adj[:k_mrr]) if a), 0.0)
    return {"P": p, "R": r, "F2": f2, "MRR@5": mrr, "hits": hits,
            "n_pred": len(pred), "n_gold": len(gold)}


# ----------------------------------------------------------------- macro
METRICS = ("P", "R", "F2", "MRR@5")


def evaluate(pred_json, gold_json, chunk_match: str = "exact", overlap_threshold: float = 0.5,
             overlap_metric: str = "f1", empty_gold: str = "skip", k_mrr: int = 5,
             per_query: bool = False) -> dict:
    """Score a prediction against gold (both: list of submission entries or a path).

    Returns ``{"doc": {P,R,F2,MRR@5,n}, "chunk": {...}, "final": mean(doc F2, chunk F2),
    "n_queries", "n_missing_pred", "n_extra_pred", "flat": {...}}`` (+ ``per_query``).
    """
    if chunk_match not in CHUNK_MATCH_MODES:
        raise ValueError(f"chunk_match must be one of {CHUNK_MATCH_MODES}")
    if empty_gold not in ("skip", "strict"):
        raise ValueError("empty_gold must be 'skip' or 'strict'")
    pred = _by_id(load_json_any(pred_json))
    gold = _by_id(load_json_any(gold_json))

    sums = {lvl: dict.fromkeys(METRICS, 0.0) for lvl in ("doc", "chunk")}
    counts = {"doc": 0, "chunk": 0}
    rows = []
    for qid in sorted(gold):
        g, pr = gold[qid], pred.get(qid, {})
        row = {"id": qid}
        gd, pd_ = g.get("relevant_docs") or [], pr.get("relevant_docs") or []
        gc, pc = g.get("relevant_chunks") or [], pr.get("relevant_chunks") or []
        levels = {
            "doc": (gd, lambda: score_docs(pd_, gd, k_mrr), len(_dedupe_docs(pd_))),
            "chunk": (gc, lambda: score_chunks(pc, gc, chunk_match, overlap_threshold,
                                               overlap_metric, k_mrr), len(_dedupe_chunks(pc))),
        }
        for lvl, (glist, fn, n_pred) in levels.items():
            if not glist:
                if empty_gold == "skip":
                    row[lvl] = None
                    continue
                s = _empty_gold_scores(n_pred)
            else:
                s = fn()
            row[lvl] = s
            counts[lvl] += 1
            for m in METRICS:
                sums[lvl][m] += s[m]
        rows.append(row)

    res: dict[str, Any] = {}
    for lvl in ("doc", "chunk"):
        n = counts[lvl]
        res[lvl] = {m: (sums[lvl][m] / n if n else 0.0) for m in METRICS}
        res[lvl]["n"] = n
    res["final"] = (res["doc"]["F2"] + res["chunk"]["F2"]) / 2
    res["n_queries"] = len(gold)
    res["n_missing_pred"] = sum(1 for q in gold if q not in pred)
    res["n_extra_pred"] = sum(1 for q in pred if q not in gold)
    res["chunk_match"] = chunk_match
    res["flat"] = {f"{lvl}_{m}": res[lvl][m] for lvl in ("doc", "chunk") for m in METRICS}
    if per_query:
        res["per_query"] = rows
    return res


def format_report(res: dict) -> str:
    lines = [f"queries={res['n_queries']} missing_pred={res['n_missing_pred']} "
             f"extra_pred={res['n_extra_pred']} chunk_match={res['chunk_match']}"]
    for lvl in ("doc", "chunk"):
        r = res[lvl]
        lines.append(f"{lvl:5s} n={r['n']:5d}  P={r['P']:.4f}  R={r['R']:.4f}  "
                     f"F2={r['F2']:.4f}  MRR@5={r['MRR@5']:.4f}")
    lines.append(f"final (mean F2) = {res['final']:.4f}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R2AI2026 offline evaluator")
    ap.add_argument("--pred", required=True, help="prediction .json/.jsonl/.zip")
    ap.add_argument("--gold", required=True, help="gold .json/.jsonl/.zip (submission format)")
    ap.add_argument("--chunk-match", default="exact", choices=CHUNK_MATCH_MODES + ("all",))
    ap.add_argument("--overlap-threshold", type=float, default=0.5)
    ap.add_argument("--overlap-metric", default="f1", choices=("f1", "recall", "jaccard"))
    ap.add_argument("--empty-gold", default="skip", choices=("skip", "strict"))
    ap.add_argument("--per-query", help="write per-query scores to this jsonl")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    a = ap.parse_args(argv)
    modes = CHUNK_MATCH_MODES if a.chunk_match == "all" else (a.chunk_match,)
    pred, gold = load_json_any(a.pred), load_json_any(a.gold)
    out = {}
    for mode in modes:
        res = evaluate(pred, gold, mode, a.overlap_threshold, a.overlap_metric, a.empty_gold,
                       per_query=bool(a.per_query))
        if a.per_query:
            path = a.per_query if len(modes) == 1 else f"{a.per_query}.{mode}"
            with open(path, "w", encoding="utf-8") as f:
                for row in res.pop("per_query"):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out[mode] = res
        if not a.json:
            print(format_report(res))
    if a.json:
        print(json.dumps(out if len(modes) > 1 else out[modes[0]], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
