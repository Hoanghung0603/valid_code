"""bge-reranker-v2-m3 served by FPT AI Marketplace (open weights, BAAI; hosted inference only).

  POST {FPT_API_BASE}/v1/rerank   Authorization: Bearer {FPT_API_KEY}
  {"model": "bge-reranker-v2-m3", "query": str, "documents": [str], "top_n": int}
  -> {"results": [{"index": i, "relevance_score": p}], "usage"|"meta": {...tokens}}
relevance_score is sigmoid(logit) in [0, 1]. Scores are cached per (model, query, document), requests are
batched (`batch_docs` documents each) and sent with a few threads; 429 / 5xx / timeouts are retried with
exponential backoff. Credentials come from .env and are never logged.
"""

from __future__ import annotations

import json
import os
import struct
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from medret.cache import KVCache, cache_key
from medret.config import PROJECT_ROOT

MODEL = "bge-reranker-v2-m3"
RETRY_CODES = (408, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524)  # 52x: Cloudflare in front of the API


def load_env(path: Path = PROJECT_ROOT / ".env") -> dict[str, str]:
    env = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return {**env, **{k: v for k, v in os.environ.items() if k.startswith("FPT_")}}


@dataclass
class RerankStats:
    requests: int = 0
    retries: int = 0
    tokens: int = 0
    cached: int = 0
    seconds: float = 0.0
    latencies: list[float] = field(default_factory=list)


class FPTReranker:
    def __init__(self, cache_path: Path | None = None, batch_docs: int = 32, threads: int = 4, timeout: float = 120,
                 max_retries: int = 6, max_doc_chars: int = 4000):
        env = load_env()
        self.base, self.key = env["FPT_API_BASE"].rstrip("/"), env["FPT_API_KEY"]
        self.batch_docs, self.threads, self.timeout, self.max_retries = batch_docs, threads, timeout, max_retries
        self.max_doc_chars = max_doc_chars  # model input is 8192 tokens; keep requests small and predictable
        self.cache = KVCache(cache_path) if cache_path else None
        self.stats = RerankStats()

    def _post(self, query: str, docs: list[str]) -> list[float]:
        body = json.dumps({"model": MODEL, "query": query, "documents": docs, "top_n": len(docs)}).encode("utf-8")
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(f"{self.base}/v1/rerank", data=body, method="POST", headers={
                "Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    res = json.loads(r.read().decode("utf-8"))
                self.stats.latencies.append(time.time() - t0)
                self.stats.requests += 1
                usage = res.get("usage") or res.get("meta", {}).get("billed_units", {})
                self.stats.tokens += int(usage.get("total_tokens", 0))
                scores = [0.0] * len(docs)
                for item in res["results"]:
                    scores[item["index"]] = float(item["relevance_score"])
                return scores
            except urllib.error.HTTPError as e:
                if e.code not in RETRY_CODES or attempt == self.max_retries:
                    raise RuntimeError(f"rerank HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:300]}") from e
            except (urllib.error.URLError, TimeoutError):
                if attempt == self.max_retries:
                    raise
            self.stats.retries += 1
            time.sleep(min(60, 2**attempt))
        raise AssertionError("unreachable")

    def score(self, query: str, docs: list[str]) -> list[float]:
        return self.score_pairs([(query, d) for d in docs])

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Relevance in [0, 1] for each (query, document) pair, in input order."""
        t0 = time.time()
        pairs = [(q, d[: self.max_doc_chars]) for q, d in pairs]
        keys = [cache_key("rerank", MODEL, q, d) for q, d in pairs]
        hit = self.cache.get_many(keys) if self.cache else {}
        scores: list[float | None] = [struct.unpack("f", hit[k])[0] if k in hit else None for k in keys]
        self.stats.cached += sum(s is not None for s in scores)
        # one request = one query and up to batch_docs of its documents
        todo: dict[str, list[int]] = {}
        for i, (q, _) in enumerate(pairs):
            if scores[i] is None:
                todo.setdefault(q, []).append(i)
        jobs = [(q, idx[s : s + self.batch_docs]) for q, idx in todo.items() for s in range(0, len(idx), self.batch_docs)]
        # results are cached as each request returns, so a failed run resumes where it stopped
        new: list[tuple[str, bytes]] = []
        with ThreadPoolExecutor(max_workers=self.threads) as pool:
            futs = {pool.submit(self._post, q, [pairs[i][1] for i in idx]): idx for q, idx in jobs}
            for f in as_completed(futs):
                idx = futs[f]
                for i, sc in zip(idx, f.result(), strict=True):
                    scores[i] = sc
                    new.append((keys[i], struct.pack("f", sc)))
                if self.cache and len(new) >= 2000:
                    self.cache.put_many(new)
                    new = []
        if self.cache and new:
            self.cache.put_many(new)
        self.stats.seconds += time.time() - t0
        return [float(s) for s in scores]
