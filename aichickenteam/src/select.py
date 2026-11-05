"""Dynamic per-query cutoff for doc / chunk selection (optimises F2).

``select(candidates, cfg) -> (docs, chunks)``

Candidates follow docs/INTERFACES.md (``chunk_id, doc_id, lang, source, text,
score, stage``; optional ``start``/``end`` offsets). Steps:

1. Drop candidates with empty text; optionally add ``lang_bias[lang]`` and
   multiply by ``lang_weights[lang]`` (applied after ``score_norm``).
2. Sort by score; dedupe chunks of the same doc that overlap (offsets if both
   have them, else text containment / identical normalised text). The
   higher-scored one is kept.
3. Chunk cutoff: keep ``s >= tau_abs`` AND ``s >= top1 - (1-alpha)*|top1|``
   (== ``alpha*top1`` for a positive top1, still sensible for negative logits),
   then clamp to ``[min_k, max_k]`` (``min_k`` wins over the thresholds, so the
   result is never empty when there are candidates). ``max_chunks_per_doc``
   optionally caps chunks per doc.
4. Doc score = aggregation over that doc's chunk scores: ``max`` | ``max_bonus``
   (max + ``doc_bonus`` * (#other chunks of the doc passing the chunk thresholds),
   bonus capped by ``doc_bonus_cap``) | ``sum_top2``. Same cutoff scheme with the
   ``doc_*`` parameters (default: the chunk ones).
5. ``relevant_docs`` = docs of the selected chunks (always) ∪ docs passing the doc
   cutoff, ordered by doc score.

``tune(dev_candidates, gold, grid, base_cfg)`` grid-searches cfg values on a dev
set and returns the best cfg (by the chosen F2 objective).

CLI::

    python -m src.select tune --cands data/cache/run/rerank.jsonl --gold gold.json \
        --grid '{"alpha":[0.5,0.7,0.9],"max_k":[3,5,10]}' --chunk-match contain
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from typing import Any, Iterable

from src.evaluate import evaluate, norm_text

DEFAULTS: dict[str, Any] = {
    "score_norm": "none",       # none | sigmoid | minmax (per query)
    "tau_abs": None,            # absolute threshold (None = disabled)
    "alpha": 0.8,               # relative threshold w.r.t. top1
    "min_k": 1,
    "max_k": 10,
    "max_chunks_per_doc": None,
    "dedupe": True,
    "dedupe_min_overlap": 0.0,  # overlap fraction (of the shorter span) above which = duplicate
    "doc_agg": "max",           # max | max_bonus | sum_top2
    "doc_bonus": 0.0,
    "doc_bonus_cap": None,
    "doc_tau_abs": "inherit",
    "doc_alpha": "inherit",
    "doc_min_k": "inherit",
    "doc_max_k": "inherit",
    "lang_bias": {},
    "lang_weights": {},
}


def resolve_cfg(cfg: dict | None) -> dict:
    c = dict(DEFAULTS)
    if cfg:
        c.update({k: v for k, v in cfg.items() if k in DEFAULTS or k.startswith("doc_")})
    for k in ("tau_abs", "alpha", "min_k", "max_k"):
        if c.get("doc_" + k) == "inherit":
            c["doc_" + k] = c[k]
    return c


def _norm_scores(scores: list[float], mode: str) -> list[float]:
    if mode in (None, "none"):
        return list(scores)
    if mode == "sigmoid":
        return [1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, s)))) for s in scores]
    if mode == "minmax":
        lo, hi = min(scores), max(scores)
        return [1.0 if hi == lo else (s - lo) / (hi - lo) for s in scores]
    raise ValueError(f"unknown score_norm {mode!r}")


def _rel_threshold(top1: float, alpha: float | None) -> float:
    if alpha is None:
        return -math.inf
    return top1 - (1.0 - alpha) * abs(top1)


def _cutoff(scored: list[tuple[float, Any]], tau_abs, alpha, min_k, max_k) -> list[Any]:
    """scored is sorted desc. Returns items passing the thresholds, clamped to [min_k, max_k]."""
    if not scored:
        return []
    thr = max(-math.inf if tau_abs is None else float(tau_abs),
              _rel_threshold(scored[0][0], alpha))
    n = sum(1 for s, _ in scored if s >= thr)
    n = max(n, int(min_k or 0))
    if max_k:
        n = min(n, int(max_k))
    return [x for _, x in scored[:max(n, 1)]]


def _overlaps(a: dict, b: dict, min_frac: float) -> bool:
    if str(a.get("doc_id")) != str(b.get("doc_id")):
        return False
    sa, ea, sb, eb = a.get("start"), a.get("end"), b.get("start"), b.get("end")
    if None not in (sa, ea, sb, eb):
        inter = min(ea, eb) - max(sa, sb)
        if inter <= 0:
            return False
        shorter = max(1, min(ea - sa, eb - sb))
        return inter / shorter > min_frac
    ta, tb = norm_text(a.get("text") or ""), norm_text(b.get("text") or "")
    return bool(ta and tb and (ta in tb or tb in ta))


def prepare(candidates: Iterable[dict], cfg: dict) -> list[dict]:
    """Normalise + weight scores, sort desc, dedupe. Returns new dicts with 'score'."""
    cands = [c for c in (candidates or []) if c and (c.get("text") or "").strip()
             and c.get("doc_id") not in (None, "")]
    if not cands:
        return []
    raw = [float(c.get("score") if c.get("score") is not None else 0.0) for c in cands]
    raw = [s if math.isfinite(s) else -1e9 for s in raw]
    normed = _norm_scores(raw, cfg["score_norm"])
    out = []
    lb, lw = cfg.get("lang_bias") or {}, cfg.get("lang_weights") or {}
    for c, s in zip(cands, normed):
        lang = c.get("lang")
        s = (s + float(lb.get(lang, 0.0))) * float(lw.get(lang, 1.0))
        d = dict(c)
        d["score"] = s
        d["doc_id"] = str(c["doc_id"])
        out.append(d)
    out.sort(key=lambda c: c["score"], reverse=True)
    if cfg.get("dedupe", True):
        kept: list[dict] = []
        by_doc: dict[str, list[dict]] = {}
        for c in out:
            if any(_overlaps(c, k, cfg["dedupe_min_overlap"]) for k in by_doc.get(c["doc_id"], [])):
                continue
            kept.append(c)
            by_doc.setdefault(c["doc_id"], []).append(c)
        out = kept
    return out


def _doc_scores(prepared: list[dict], cfg: dict, chunk_thr: float) -> list[tuple[float, str]]:
    per_doc: dict[str, list[float]] = {}
    for c in prepared:  # already sorted desc
        per_doc.setdefault(c["doc_id"], []).append(c["score"])
    agg = cfg["doc_agg"]
    res = []
    for d, ss in per_doc.items():
        if agg == "max":
            v = ss[0]
        elif agg == "sum_top2":
            v = sum(ss[:2])
        elif agg == "max_bonus":
            n_good = sum(1 for s in ss[1:] if s >= chunk_thr)
            bonus = float(cfg["doc_bonus"]) * n_good
            if cfg.get("doc_bonus_cap") is not None:
                bonus = min(bonus, float(cfg["doc_bonus_cap"]))
            v = ss[0] + bonus
        else:
            raise ValueError(f"unknown doc_agg {agg!r}")
        res.append((v, d))
    res.sort(key=lambda x: x[0], reverse=True)
    return res


def select(candidates: list[dict], cfg: dict | None = None) -> tuple[list[str], list[dict]]:
    """Return (relevant_docs, relevant_chunks[{doc_id, chunk_text}]) for one query."""
    c = resolve_cfg(cfg)
    prepared = prepare(candidates, c)
    if not prepared:
        return [], []
    scored = [(x["score"], x) for x in prepared]
    if c.get("max_chunks_per_doc"):
        cap, cnt, capped = int(c["max_chunks_per_doc"]), {}, []
        for s, x in scored:
            cnt[x["doc_id"]] = cnt.get(x["doc_id"], 0) + 1
            if cnt[x["doc_id"]] <= cap:
                capped.append((s, x))
        scored = capped
    chunks = _cutoff(scored, c["tau_abs"], c["alpha"], c["min_k"], c["max_k"])
    chunk_thr = max(-math.inf if c["tau_abs"] is None else float(c["tau_abs"]),
                    _rel_threshold(scored[0][0], c["alpha"]))

    doc_scored = _doc_scores(prepared, c, chunk_thr)
    docs_sel = set(_cutoff(doc_scored, c["doc_tau_abs"], c["doc_alpha"],
                           c["doc_min_k"], c["doc_max_k"]))
    docs_sel |= {x["doc_id"] for x in chunks}
    docs = [d for _, d in doc_scored if d in docs_sel]
    return docs, [{"doc_id": x["doc_id"], "chunk_text": x["text"]} for x in chunks]


# ------------------------------------------------------------------- tuning
def apply(dev_candidates: dict, cfg: dict, query_ids: Iterable[int] | None = None) -> list[dict]:
    ids = list(query_ids) if query_ids is not None else list(dev_candidates)
    out = []
    for qid in ids:
        docs, chunks = select(dev_candidates.get(qid) or [], cfg)
        out.append({"id": int(qid), "relevant_docs": docs, "relevant_chunks": chunks})
    return out


def _objective(res: dict, objective: str) -> float:
    if objective == "doc_F2":
        return res["doc"]["F2"]
    if objective == "chunk_F2":
        return res["chunk"]["F2"]
    if objective == "mean_F2":
        return res["final"]
    raise ValueError(f"unknown objective {objective!r}")


def tune(dev_candidates: dict, gold: list[dict], grid: dict[str, list], base_cfg: dict | None = None,
         objective: str = "mean_F2", chunk_match: str = "contain", eval_kwargs: dict | None = None,
         verbose: bool = False) -> tuple[dict, float, list[dict]]:
    """Grid search. dev_candidates: {qid: [Candidate]}; gold: submission-format list.

    Returns (best_cfg, best_score, all_results sorted desc). Each result has
    ``cfg`` (the grid point), ``score`` and ``metrics`` (flat evaluate output).
    """
    gold = list(gold)
    gold_ids = [int(g["id"]) for g in gold]
    dev = {int(k): v for k, v in dev_candidates.items()}
    keys = list(grid)
    results = []
    for values in itertools.product(*(grid[k] for k in keys)):
        point = dict(zip(keys, values))
        cfg = {**(base_cfg or {}), **point}
        pred = apply(dev, cfg, gold_ids)
        res = evaluate(pred, gold, chunk_match=chunk_match, **(eval_kwargs or {}))
        score = _objective(res, objective)
        results.append({"cfg": point, "score": score, "metrics": res["flat"]})
        if verbose:
            print(f"{score:.4f} {json.dumps(point)}", file=sys.stderr)
    results.sort(key=lambda r: r["score"], reverse=True)
    if not results:
        return dict(base_cfg or {}), float("nan"), []
    best = {**(base_cfg or {}), **results[0]["cfg"]}
    return best, results[0]["score"], results


def load_candidates_cache(path: str) -> dict[int, list[dict]]:
    """Read a run.py stage cache (jsonl of {"id", "data": [Candidate]})."""
    out: dict[int, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                out[int(rec["id"])] = rec.get("data") or []
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="select / tune dynamic cutoffs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tune")
    t.add_argument("--cands", required=True, help="jsonl cache of reranked candidates")
    t.add_argument("--gold", required=True)
    t.add_argument("--grid", required=True, help="JSON dict or path to JSON/YAML file")
    t.add_argument("--config", default="configs/default.yaml", help="base cfg (select: block)")
    t.add_argument("--objective", default="mean_F2", choices=("mean_F2", "doc_F2", "chunk_F2"))
    t.add_argument("--chunk-match", default="contain")
    t.add_argument("--top", type=int, default=10)
    t.add_argument("--out", help="write all results as json")
    a = ap.parse_args(argv)

    from src.evaluate import load_json_any
    base = {}
    try:
        import yaml  # type: ignore
        with open(a.config, encoding="utf-8") as f:
            base = (yaml.safe_load(f) or {}).get("select") or {}
    except Exception as e:  # noqa: BLE001
        print(f"[select] no base config ({e}); using defaults", file=sys.stderr)
    if a.grid.strip().startswith("{"):
        grid = json.loads(a.grid)
    else:
        with open(a.grid, encoding="utf-8") as f:
            txt = f.read()
        try:
            grid = json.loads(txt)
        except json.JSONDecodeError:
            import yaml  # type: ignore
            grid = yaml.safe_load(txt)
    best, score, results = tune(load_candidates_cache(a.cands), load_json_any(a.gold), grid,
                                base, a.objective, a.chunk_match)
    for r in results[:a.top]:
        m = r["metrics"]
        print(f"{r['score']:.4f}  docF2={m['doc_F2']:.4f} chunkF2={m['chunk_F2']:.4f}  "
              f"{json.dumps(r['cfg'], ensure_ascii=False)}")
    print("best:", json.dumps(best, ensure_ascii=False))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump({"best": best, "score": score, "results": results}, f, ensure_ascii=False,
                      indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
