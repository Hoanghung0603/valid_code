"""Local re-implementation of the official scorer (docs/about-competition/evaluation.md).

Spec of 2026-10-10: precision = relevant predicted chunks / all predicted chunks (no near-duplicate merging any
more), recall = reference chunks found / all reference chunks, a chunk matches when LCS_tokens(c, g) >= 0.4 |g|
after normalisation (normalize.py), macro over queries with >= 1 positive at that level.
Rule change announced by the organisers (applied on the LB between 2026-10-21 and 10-24): a match also needs
LCS_tokens(c, g) >= 0.4 |c|, so a predicted chunk more than 2.5x longer than the reference no longer matches
(`pred_threshold`; None restores the earlier rule). The same whole-document chunks fell from chunk P/R 0.271/0.458 to
0.225/0.357 on the LB.
Choices where the spec is silent (see docs/analysis/competition-analysis.md, section 3):
- The token unit is not specified; default whitespace tokens of the normalised text (config eval.tokenizer),
  a subword tokenizer can be passed instead.
- A reference chunk is found when some predicted chunk of the same doc matches it.
- `near_duplicate_threshold` (merging of predictions, from the earlier spec) is kept as an option, off by default.
- Doc ids are compared as strings, so int and str ids are interchangeable here.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Sequence
from dataclasses import asdict, dataclass, field

from medret.eval.lcs import LCSMatcher, lcs_length
from medret.eval.normalize import normalize_text
from medret.eval.tokenize import TokenizeFn
from medret.schemas import QueryResult


def fbeta(p: float, r: float, beta: float = 2.0) -> float:
    b2 = beta * beta
    denom = b2 * p + r
    return (1 + b2) * p * r / denom if denom > 0 else 0.0


@dataclass
class LevelScore:
    precision: float
    recall: float
    f: float


@dataclass
class QueryScore:
    id: int
    doc: LevelScore | None  # None: query has no positive at this level -> excluded from the macro
    chunk: LevelScore | None
    n_pred_docs: int = 0
    n_gold_docs: int = 0
    n_pred_chunks: int = 0  # after near-duplicate merging when that option is on
    n_ref_chunks: int = 0


@dataclass
class EvalReport:
    doc_f2: float
    chunk_f2: float
    final: float
    doc_precision: float
    doc_recall: float
    chunk_precision: float
    chunk_recall: float
    n_doc_queries: int
    n_chunk_queries: int
    per_query: list[QueryScore] = field(default_factory=list)

    def summary(self) -> dict:
        d = asdict(self)
        d.pop("per_query")
        return d


def score_docs(pred: Sequence[str], gold: Sequence[str], beta: float) -> LevelScore | None:
    g = set(gold)
    if not g:
        return None
    d = set(pred)
    hit = len(d & g)
    p = hit / len(d) if d else 0.0
    r = hit / len(g)
    return LevelScore(p, r, fbeta(p, r, beta))


def score_chunks(
    pred: Sequence[tuple[str, Sequence[Hashable]]],
    refs: Sequence[tuple[str, Sequence[Hashable]]],
    relevance_threshold: float,
    near_duplicate_threshold: float | None,
    beta: float,
    pred_threshold: float | None = 0.4,
) -> tuple[LevelScore | None, int]:
    refs = [(doc, toks) for doc, toks in refs if toks]
    if not refs:
        return None, 0

    refs_by_doc: dict[str, list[int]] = defaultdict(list)
    for i, (doc, _) in enumerate(refs):
        refs_by_doc[doc].append(i)

    covered: set[int] = set()
    groups: dict[str, list[tuple[Sequence[Hashable], bool]]] = defaultdict(list)  # doc -> [(repr, relevant)]
    for doc, toks in pred:
        matcher = LCSMatcher(toks)
        hits = []
        for i in refs_by_doc.get(doc, ()):
            common = matcher.lcs(refs[i][1])
            if common >= relevance_threshold * len(refs[i][1]) and (
                    pred_threshold is None or common >= pred_threshold * len(toks)):
                hits.append(i)
        covered.update(hits)
        relevant = bool(hits)

        doc_groups = groups[doc]
        if near_duplicate_threshold is None:
            doc_groups.append((toks, relevant))
            continue
        for gi, (rep, rep_relevant) in enumerate(doc_groups):
            common = lcs_length(rep, toks)
            union = len(rep) + len(toks) - common
            if union == 0 or common / union >= near_duplicate_threshold:
                doc_groups[gi] = (rep, rep_relevant or relevant)
                break
        else:
            doc_groups.append((toks, relevant))

    all_groups = [g for gs in groups.values() for g in gs]
    n_pred = len(all_groups)
    p = sum(rel for _, rel in all_groups) / n_pred if n_pred else 0.0
    r = len(covered) / len(refs)
    return LevelScore(p, r, fbeta(p, r, beta)), n_pred


def evaluate(
    predictions: Sequence[QueryResult],
    qrels: Sequence[QueryResult],
    tokenize: TokenizeFn,
    relevance_threshold: float = 0.4,
    near_duplicate_threshold: float | None = None,
    beta: float = 2.0,
    pred_threshold: float | None = 0.4,
) -> EvalReport:
    """Queries present in qrels but missing from predictions score as empty predictions."""
    preds_by_id = {q.id: q for q in predictions}
    per_query: list[QueryScore] = []

    for gold in qrels:
        pred = preds_by_id.get(gold.id, QueryResult(id=gold.id))
        texts = [normalize_text(c.chunk_text) for c in pred.relevant_chunks + gold.relevant_chunks]
        toks = tokenize(texts) if texts else []
        n = len(pred.relevant_chunks)
        pred_chunks = [(c.doc_id, t) for c, t in zip(pred.relevant_chunks, toks[:n], strict=True)]
        ref_chunks = [(c.doc_id, t) for c, t in zip(gold.relevant_chunks, toks[n:], strict=True)]

        chunk, n_pred_chunks = score_chunks(
            pred_chunks, ref_chunks, relevance_threshold, near_duplicate_threshold, beta, pred_threshold
        )
        per_query.append(
            QueryScore(
                id=gold.id,
                doc=score_docs(pred.relevant_docs, gold.relevant_docs, beta),
                chunk=chunk,
                n_pred_docs=len(set(pred.relevant_docs)),
                n_gold_docs=len(set(gold.relevant_docs)),
                n_pred_chunks=n_pred_chunks,
                n_ref_chunks=len(gold.relevant_chunks),
            )
        )

    def macro(level: str, attr: str) -> tuple[float, int]:
        vals = [getattr(getattr(q, level), attr) for q in per_query if getattr(q, level) is not None]
        return (sum(vals) / len(vals) if vals else 0.0), len(vals)

    doc_f2, n_doc = macro("doc", "f")
    chunk_f2, n_chunk = macro("chunk", "f")
    levels = [s for s, n in ((doc_f2, n_doc), (chunk_f2, n_chunk)) if n]
    return EvalReport(
        doc_f2=doc_f2,
        chunk_f2=chunk_f2,
        final=sum(levels) / len(levels) if levels else 0.0,
        doc_precision=macro("doc", "precision")[0],
        doc_recall=macro("doc", "recall")[0],
        chunk_precision=macro("chunk", "precision")[0],
        chunk_recall=macro("chunk", "recall")[0],
        n_doc_queries=n_doc,
        n_chunk_queries=n_chunk,
        per_query=per_query,
    )
