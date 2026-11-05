#!/usr/bin/env python
"""End-to-end R2AI2026 pipeline.

    python run.py --queries data/ViBioMIR/query.parquet --out submission.zip
    python run.py --stages select,submit --out sub_tuned.zip      # reuse cached rerank
    python run.py --limit 20 --gold dev_gold.json --out dev.zip    # quick dev run + eval

Stages (executed stage-major, i.e. every query goes through a stage before the next
stage starts, so only one big model is resident at a time):

    plan     src.query.build_plans(queries)                      -> QueryPlan
    pubmed   src.pubmed.candidates_for_plan(plan)                 -> [Doc]
    retrieve src.retrieve.retrieve(plan, k, <extra docs>)         -> [Candidate]
    rerank   src.rerank.rerank(query_vi, candidates)              -> [Candidate]
    select   src.select.select(candidates, cfg["select"])         -> docs, chunks
    submit   complete + validate + write_zip

Caching: each stage appends ``{"id", "in", "data"}`` records to
``<cache_dir>/<stage>.jsonl`` (``in`` = hash of the stage input + stage config). A stage
listed in ``--stages`` reuses records whose ``in`` hash matches (resume) and recomputes
the rest (``--force`` recomputes everything); a stage *not* listed is read from cache
only (missing -> empty). Any per-query failure yields an empty result for that query, so
the submission always contains every id.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import inspect
import json
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

log = logging.getLogger("run")

STAGES = ("plan", "pubmed", "retrieve", "rerank", "select", "submit")
STAGE_MODULE = {"plan": "src.query", "pubmed": "src.pubmed", "retrieve": "src.retrieve",
                "rerank": "src.rerank"}
RUN_DEFAULTS: dict[str, Any] = {
    "cache_dir": "data/cache/run",
    "retrieve_k": 200,
    "use_pubmed": True,
    "pubmed_cache": "full",       # full | ids (ids: re-fetch via src.pubmed.fetch)
    "pubmed_max_docs": 300,
    "pubmed_max_chunks": 400,     # only when retrieve() cannot take extra docs
    "rerank_max_cands": 300,
    "cache_rerank_top": 200,      # candidates kept in rerank cache (for select tuning)
    "json_name": "submission.json",
    "time_budget_s": None,
    "log_every": 100,
}


class StageUnavailable(RuntimeError):
    pass


# ----------------------------------------------------------------- utilities
def load_config(path: str | None) -> dict:
    p = Path(path) if path else ROOT / "configs" / "default.yaml"
    if not p.exists():
        log.warning("config %s not found, using defaults", p)
        return {}
    import yaml
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def load_queries(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".parquet":
        import pyarrow.parquet as pq
        t = pq.read_table(p, columns=["id", "query"])
        rows = [{"id": int(i), "query": q} for i, q in
                zip(t.column("id").to_pylist(), t.column("query").to_pylist())]
    elif p.suffix == ".jsonl":
        rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        rows = json.loads(p.read_text(encoding="utf-8"))
    return [{"id": int(r["id"]), "query": str(r["query"] or "")} for r in rows]


def jhash(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, ensure_ascii=False, sort_keys=True, default=str)
                        .encode("utf-8")).hexdigest()[:16]


def call_flex(fn: Callable, *args, **maybe_kwargs):
    """Call fn(*args, **kw) passing only the kwargs fn accepts (all if it takes **kw)."""
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args)
    params = sig.parameters
    if any(p.kind == p.VAR_KEYWORD for p in params.values()):
        kw = maybe_kwargs
    else:
        kw = {k: v for k, v in maybe_kwargs.items() if k in params}
    return fn(*args, **kw)


def accepts(fn: Callable, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def import_stage(stage: str):
    mod = STAGE_MODULE[stage]
    try:
        return importlib.import_module(mod)
    except Exception as e:  # noqa: BLE001
        raise StageUnavailable(f"stage '{stage}' needs {mod}, which failed to import: "
                               f"{type(e).__name__}: {e}") from e


def fallback_plan(q: dict) -> dict:
    return {"id": q["id"], "query_vi": q["query"], "vi": [q["query"]], "en": [], "zh": [],
            "pubmed": [], "keywords": {"vi": [], "en": [], "zh": []}}


class StageCache:
    def __init__(self, cache_dir: Path, stage: str):
        self.path = cache_dir / f"{stage}.jsonl"
        self.recs: dict[int, dict] = {}
        if self.path.exists():
            with self.path.open(encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:  # torn last line after a crash
                        continue
                    self.recs[int(r["id"])] = r
        self._fh = None

    def get(self, qid: int, in_hash: str | None = None):
        r = self.recs.get(qid)
        if r is None or (in_hash is not None and r.get("in") != in_hash):
            return None
        return r

    def put(self, qid: int, in_hash: str, data: Any, err: str | None = None) -> None:
        rec = {"id": qid, "in": in_hash, "data": data}
        if err:
            rec["err"] = err
        self.recs[qid] = rec
        if self._fh is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = self.path.open("a", encoding="utf-8")
        self._fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self, compact: bool = True) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            if compact:
                tmp = self.path.with_suffix(".jsonl.tmp")
                with tmp.open("w", encoding="utf-8") as f:
                    for qid in sorted(self.recs):
                        f.write(json.dumps(self.recs[qid], ensure_ascii=False) + "\n")
                tmp.replace(self.path)


class Timer:
    def __init__(self):
        self.stats: dict[str, dict] = {}

    def record(self, stage: str, **kw) -> None:
        self.stats.setdefault(stage, {}).update(kw)

    def summary(self) -> str:
        lines = [f"{'stage':9s} {'secs':>9s} {'s/query':>8s} {'computed':>8s} {'cached':>7s} "
                 f"{'failed':>6s}"]
        for st, s in self.stats.items():
            n = max(1, s.get("computed", 0))
            lines.append(f"{st:9s} {s.get('secs', 0):9.1f} {s.get('secs', 0) / n:8.3f} "
                         f"{s.get('computed', 0):8d} {s.get('cached', 0):7d} {s.get('failed', 0):6d}")
        return "\n".join(lines)


# ---------------------------------------------------------------- pipeline
class Pipeline:
    def __init__(self, cfg: dict, stages: list[str], cache_dir: Path, force: bool = False,
                 allow_missing: bool = False):
        self.cfg = cfg
        self.run_cfg = {**RUN_DEFAULTS, **(cfg.get("run") or {})}
        self.stages = stages
        self.cache_dir = cache_dir
        self.force = force
        self.allow_missing = allow_missing
        self.timer = Timer()
        self.mods: dict[str, Any] = {}
        self.t_start = time.time()

    # -- module checks
    def check_modules(self) -> None:
        problems = []
        for st in self.stages:
            if st not in STAGE_MODULE or (st == "pubmed" and not self.run_cfg["use_pubmed"]):
                continue
            try:
                self.mods[st] = import_stage(st)
            except StageUnavailable as e:
                problems.append(str(e))
                self.mods[st] = None
        if problems:
            msg = "\n  ".join(problems)
            if not self.allow_missing:
                raise StageUnavailable(
                    f"missing modules:\n  {msg}\n(use --allow-missing to run those stages as "
                    "pass-through, or drop them from --stages to read their cache)")
            log.warning("running with pass-through stages:\n  %s", msg)

    # -- generic per-query stage driver
    def run_stage(self, stage: str, queries: list[dict], inputs: dict[int, Any],
                  fn: Callable[[dict, Any], Any], empty: Any, cfg_part: Any = None) -> dict[int, Any]:
        cache = StageCache(self.cache_dir, stage)
        out: dict[int, Any] = {}
        active = stage in self.stages
        t0 = time.time()
        n_cached = n_comp = n_fail = n_stale = 0
        every = int(self.run_cfg["log_every"] or 0)
        passthrough = stage in STAGE_MODULE and self.mods.get(stage) is None
        for i, q in enumerate(queries):
            qid = q["id"]
            in_hash = jhash([q["query"], inputs.get(qid), cfg_part])
            if not active:
                r = cache.get(qid)
                if r is None:
                    out[qid] = empty
                    n_fail += 1
                else:
                    n_stale += r.get("in") != in_hash
                    out[qid] = r["data"]
                    n_cached += 1
                continue
            r = None if self.force else cache.get(qid, in_hash)
            if r is not None and not r.get("err"):  # failed/pass-through records are retried
                out[qid] = r["data"]
                n_cached += 1
                continue
            err = None
            try:
                data = fn(q, inputs.get(qid))
            except Exception as e:  # noqa: BLE001
                log.error("[%s] query %s failed: %s", stage, qid, e)
                log.debug(traceback.format_exc())
                data, err = empty, f"{type(e).__name__}: {e}"
                n_fail += 1
            if err is None and passthrough:
                err = "passthrough"
            out[qid] = data
            cache.put(qid, in_hash, data, err)
            n_comp += 1
            if every and n_comp % every == 0:
                el = time.time() - t0
                log.info("[%s] %d/%d (%.2fs/query, ETA %.1f min)", stage, i + 1, len(queries),
                         el / n_comp, el / n_comp * (len(queries) - i - 1) / 60)
        cache.close()
        if not active and (n_fail or n_stale):
            log.warning("[%s] read from cache: %d missing (empty), %d stale vs current inputs",
                        stage, n_fail, n_stale)
        self.timer.record(stage, secs=time.time() - t0, computed=n_comp, cached=n_cached,
                          failed=n_fail)
        self._budget_check(stage)
        return out

    def _budget_check(self, stage: str) -> None:
        el = time.time() - self.t_start
        s = self.timer.stats[stage]
        log.info("[%s] done in %.1fs (computed=%d cached=%d failed=%d); total elapsed %.1fs",
                 stage, s["secs"], s["computed"], s["cached"], s["failed"], el)
        budget = self.run_cfg.get("time_budget_s")
        if budget and el > float(budget):
            log.warning("time budget exceeded: %.1fs > %ss after stage %s", el, budget, stage)

    # -- stages
    def stage_plan(self, queries: list[dict]) -> dict[int, dict]:
        mod = self.mods.get("plan")
        cfg_part = self.cfg.get("query")
        if "plan" in self.stages and mod is not None:
            # batch call (LLM): only for queries not already cached with same input
            cache = StageCache(self.cache_dir, "plan")
            todo = [q for q in queries
                    if self.force or cache.get(q["id"], jhash([q["query"], None, cfg_part])) is None]
            batch: dict[int, Any] = {}
            if todo:
                t0 = time.time()
                try:
                    plans = mod.build_plans(todo)
                    batch = {q["id"]: p for q, p in zip(todo, plans)}
                except Exception as e:  # noqa: BLE001
                    log.error("[plan] build_plans failed (%s); falling back per query", e)
                    log.debug(traceback.format_exc())
                log.info("[plan] build_plans for %d queries took %.1fs", len(todo), time.time() - t0)

            def fn(q, _):
                p = batch.get(q["id"])
                if not p:
                    raise RuntimeError("build_plans returned no plan")
                return p
        else:
            fn = lambda q, _: fallback_plan(q)  # noqa: E731
        plans = self.run_stage("plan", queries, {}, fn, None, cfg_part)
        # empty/failed plan -> fallback so downstream still works
        return {q["id"]: (plans.get(q["id"]) or fallback_plan(q)) for q in queries}

    def stage_pubmed(self, queries, plans) -> dict[int, list[dict]]:
        mod = self.mods.get("pubmed")
        rc = self.run_cfg
        if not rc["use_pubmed"]:
            return {q["id"]: [] for q in queries}

        def fn(q, plan):
            if mod is None:
                return []
            docs = list(mod.candidates_for_plan(plan) or [])[: int(rc["pubmed_max_docs"])]
            if rc["pubmed_cache"] == "ids":
                return [str(d["doc_id"]) for d in docs]
            keep = ("doc_id", "source", "lang", "url", "title", "text", "fetched_at")
            return [{k: d.get(k) for k in keep if k in d} for d in docs]

        res = self.run_stage("pubmed", queries, plans, fn, [], self.cfg.get("pubmed"))
        if rc["pubmed_cache"] == "ids" and any(v and isinstance(v[0], str) for v in res.values()):
            mod = mod or import_stage("pubmed")
            for qid, v in res.items():
                if v and isinstance(v[0], str):
                    res[qid] = list(mod.fetch(v))
        return res

    def _pubmed_chunks(self, docs: list[dict]) -> list[dict]:
        """Fallback when retrieve() cannot take extra docs: chunk PubMed docs ourselves."""
        try:
            chunk_mod = importlib.import_module("src.chunk")
            chunk_doc = chunk_mod.chunk_doc
        except Exception:  # noqa: BLE001
            chunk_doc = None
        out = []
        for d in docs:
            if chunk_doc is not None:
                try:
                    chs = call_flex(chunk_doc, d, self.cfg.get("chunk") or {})
                except Exception:  # noqa: BLE001
                    chs = None
            else:
                chs = None
            if not chs:  # whole text as one chunk
                chs = [{"chunk_id": f"{d['doc_id']}#0", "doc_id": d["doc_id"], "start": 0,
                        "end": len(d.get("text") or ""), "text": d.get("text") or ""}]
            for c in chs:
                out.append({"chunk_id": c.get("chunk_id"), "doc_id": str(d["doc_id"]),
                            "lang": d.get("lang", "en"), "source": d.get("source", "pubmed"),
                            "text": c.get("text"), "start": c.get("start"), "end": c.get("end"),
                            "score": 0.0, "stage": "pubmed"})
        return out[: int(self.run_cfg["pubmed_max_chunks"])]

    def stage_retrieve(self, queries, plans, pub_docs) -> dict[int, list[dict]]:
        mod = self.mods.get("retrieve")
        k = int(self.run_cfg["retrieve_k"])
        inputs = {q["id"]: {"plan": plans.get(q["id"]),
                            "pubmed": [d.get("doc_id") for d in pub_docs.get(q["id"]) or []]}
                  for q in queries}
        extra_name = None
        if mod is not None:
            for name in ("extra_docs", "pubmed_docs", "docs", "extra"):
                if accepts(mod.retrieve, name):
                    extra_name = name
                    break

        def fn(q, inp):
            plan, docs = inp["plan"], pub_docs.get(q["id"]) or []
            if mod is None:
                return self._pubmed_chunks(docs)
            if extra_name:
                return list(mod.retrieve(plan, k, **{extra_name: docs}) or [])
            cands = list(mod.retrieve(plan, k) or [])
            seen = {c.get("chunk_id") for c in cands}
            return cands + [c for c in self._pubmed_chunks(docs) if c["chunk_id"] not in seen]

        return self.run_stage("retrieve", queries, inputs, fn, [],
                              [self.cfg.get("retrieve"), self.cfg.get("index"), k])

    def stage_rerank(self, queries, plans, cands) -> dict[int, list[dict]]:
        mod = self.mods.get("rerank")
        nmax = int(self.run_cfg["rerank_max_cands"])
        keep_top = int(self.run_cfg["cache_rerank_top"])
        inputs = {q["id"]: [c.get("chunk_id") for c in cands.get(q["id"]) or []] for q in queries}

        def fn(q, _):
            cs = list(cands.get(q["id"]) or [])
            if not cs:
                return []
            first = [c for c in cs if c.get("stage") != "pubmed"]
            first.sort(key=lambda c: c.get("score") or 0.0, reverse=True)
            pub = [c for c in cs if c.get("stage") == "pubmed"]
            n_pub = min(len(pub), max(nmax - len(first), nmax // 3))
            cs = first[:nmax - n_pub] + pub[:n_pub]
            if mod is None:
                out = cs
            else:
                qv = (plans.get(q["id"]) or {}).get("query_vi") or q["query"]
                out = list(call_flex(mod.rerank, qv, cs, cfg=self.cfg.get("rerank"),
                                     plan=plans.get(q["id"])) or [])
            out.sort(key=lambda c: c.get("score") if c.get("score") is not None else -1e9,
                     reverse=True)
            return out[:keep_top]

        return self.run_stage("rerank", queries, inputs, fn, [], self.cfg.get("rerank"))

    def stage_select(self, queries, cands) -> dict[int, dict]:
        from src.select import select
        scfg = self.cfg.get("select") or {}
        inputs = {q["id"]: jhash(cands.get(q["id"])) for q in queries}

        def fn(q, _):
            docs, chunks = select(cands.get(q["id"]) or [], scfg)
            return {"relevant_docs": docs, "relevant_chunks": chunks}

        return self.run_stage("select", queries, inputs, fn,
                              {"relevant_docs": [], "relevant_chunks": []}, scfg)

    def free(self, stage: str) -> None:
        mod = self.mods.get(stage)
        for name in ("unload", "release", "free"):
            f = getattr(mod, name, None) if mod is not None else None
            if callable(f):
                try:
                    f()
                except Exception:  # noqa: BLE001
                    pass
                break

    def run(self, queries: list[dict]) -> list[dict]:
        self.check_modules()
        plans = self.stage_plan(queries)
        self.free("plan")
        pub = self.stage_pubmed(queries, plans)
        cands = self.stage_retrieve(queries, plans, pub)
        self.free("retrieve")
        rer = self.stage_rerank(queries, plans, cands)
        self.free("rerank")
        sel = self.stage_select(queries, rer)
        return [{"id": q["id"], **(sel.get(q["id"]) or {"relevant_docs": [], "relevant_chunks": []})}
                for q in queries]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="R2AI2026 end-to-end pipeline")
    ap.add_argument("--queries", default=None, help="parquet/jsonl/json with id, query")
    ap.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    ap.add_argument("--out", default="submission.zip")
    ap.add_argument("--out-json", help="also write the raw submission JSON here")
    ap.add_argument("--limit", type=int, help="only the first N queries (still writes N entries)")
    ap.add_argument("--stages", default="all",
                    help=f"comma list of stages to compute (others read from cache): {','.join(STAGES)}")
    ap.add_argument("--cache-dir", help="override run.cache_dir")
    ap.add_argument("--force", action="store_true", help="ignore cache for the listed stages")
    ap.add_argument("--allow-missing", action="store_true",
                    help="run stages whose module is missing as pass-through")
    ap.add_argument("--gold", help="evaluate the result against this gold file")
    ap.add_argument("--chunk-match", default="contain")
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if a.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", force=True)

    cfg = load_config(a.config)
    stages = list(STAGES) if a.stages == "all" else [s.strip() for s in a.stages.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        ap.error(f"unknown stage(s) {bad}; valid: {STAGES}")
    qpath = a.queries or (cfg.get("paths") or {}).get("queries") or "data/ViBioMIR/query.parquet"
    queries = load_queries(qpath if Path(qpath).is_absolute() or Path(qpath).exists() else str(ROOT / qpath))
    if a.limit:
        queries = queries[: a.limit]
    run_cfg = {**RUN_DEFAULTS, **(cfg.get("run") or {})}
    cache_dir = Path(a.cache_dir or run_cfg["cache_dir"])
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir
    log.info("queries=%d stages=%s cache=%s", len(queries), stages, cache_dir)

    pipe = Pipeline(cfg, stages, cache_dir, force=a.force, allow_missing=a.allow_missing)
    try:
        pred = pipe.run(queries)
    except StageUnavailable as e:
        log.error("%s", e)
        return 2

    from src.submit import complete, validate, write_zip
    qids = [q["id"] for q in queries]
    pred = complete(pred, qids)
    errs = validate(pred, qids)
    if errs:
        log.error("submission invalid (%d errors): %s", len(errs), errs[:5])
        return 1
    n_empty = sum(1 for e in pred if not e["relevant_docs"])
    log.info("entries=%d empty=%d avg_docs=%.2f avg_chunks=%.2f", len(pred), n_empty,
             sum(len(e["relevant_docs"]) for e in pred) / max(1, len(pred)),
             sum(len(e["relevant_chunks"]) for e in pred) / max(1, len(pred)))
    if "submit" in stages:
        t0 = time.time()
        write_zip(pred, a.out, run_cfg["json_name"])
        pipe.timer.record("submit", secs=time.time() - t0, computed=len(pred), cached=0, failed=0)
        log.info("wrote %s", a.out)
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")
    print(pipe.timer.summary(), file=sys.stderr)
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "timing.json").write_text(json.dumps(
        {"total_s": time.time() - pipe.t_start, "n_queries": len(queries), "stages": pipe.timer.stats},
        indent=1), encoding="utf-8")
    if a.gold:
        from src.evaluate import evaluate, format_report
        print(format_report(evaluate(pred, a.gold, chunk_match=a.chunk_match)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
