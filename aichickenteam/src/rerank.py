"""Cross-encoder reranking (BAAI/bge-reranker-v2-m3, multilingual) with calibrated scores.

rerank(query_vi, candidates, top_n=None, cfg=None) -> list[Candidate]
  - scores (query_vi, chunk text) pairs directly (no translation needed: VI query vs VI/ZH/EN chunk)
  - fp16, length-sorted token-budget batches, OOM back-off, max_length configurable
  - candidate["score"] = sigmoid(logit) in [0, 1] (thresholdable); "logit" and the previous
    "retrieval_score" are kept; stage = "rerank"
  - persistent sqlite cache data/cache/rerank.sqlite keyed by sha1(model|max_length|query|text)
"""
from __future__ import annotations

import gc
import hashlib
import sqlite3
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

DEFAULTS = {"model": "BAAI/bge-reranker-v2-m3", "max_length": 512, "batch_tokens": 8192, "fp16": True,
            "top_n": None, "cache": True, "cache_path": "data/cache/rerank.sqlite"}


def rrcfg(cfg: dict | None) -> dict:
    """Accepts the full config (uses cfg["rerank"]) or the rerank block itself; None -> configs/default.yaml."""
    if cfg is None:
        try:
            from src.index import load_config
            cfg = load_config()
        except Exception:
            cfg = {}
    block = cfg.get("rerank") if isinstance(cfg.get("rerank"), dict) else \
        (cfg if any(k in cfg for k in DEFAULTS) else {})
    r = dict(DEFAULTS)
    r.update({k: v for k, v in (block or {}).items() if v is not None or k == "top_n"})
    return r


class Reranker:
    def __init__(self, model=DEFAULTS["model"], max_length=512, batch_tokens=8192, fp16=True, device=None,
                 cache=True, cache_path=DEFAULTS["cache_path"]):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.name = model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model)
        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = AutoModelForSequenceClassification.from_pretrained(model, dtype=dtype).to(self.device).eval()
        self.max_length = int(max_length)
        self.batch_tokens = int(batch_tokens)
        self.use_cache = cache
        p = Path(cache_path)
        self.cache_path = p if p.is_absolute() else ROOT / p
        self._db = None

    def _conn(self):
        if self._db is None:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(self.cache_path), timeout=60)
            self._db.execute("CREATE TABLE IF NOT EXISTS rr (k TEXT PRIMARY KEY, v REAL)")
        return self._db

    def _key(self, q, t):
        return hashlib.sha1(f"{self.name}|{self.max_length}|{q}\x00{t}".encode()).hexdigest()

    def logits(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        torch = self.torch
        out = np.zeros(len(pairs), dtype=np.float32)
        if not pairs:
            return out
        enc = self.tok([p[0] for p in pairs], [p[1] for p in pairs], truncation="only_second",
                       max_length=self.max_length, padding=False)
        ids = enc["input_ids"]
        lengths = np.array([len(x) for x in ids])
        order = np.argsort(-lengths, kind="stable")
        budget = self.batch_tokens
        i = 0
        while i < len(order):
            L = int(lengths[order[i]])
            bs = max(1, min(128, budget // max(L, 1)))
            idx = order[i:i + bs]
            feats = None
            try:
                feats = self.tok.pad({k: [enc[k][j] for j in idx] for k in enc.keys()}, return_tensors="pt")
                feats = {k: v.to(self.device) for k, v in feats.items()}
                with torch.inference_mode():
                    lg = self.model(**feats).logits.view(-1).float()
                out[idx] = lg.cpu().numpy()
                i += bs
            except torch.cuda.OutOfMemoryError:
                feats = None
                gc.collect()
                torch.cuda.empty_cache()
                if bs == 1:
                    raise
                budget = max(L, budget // 2)
                self.batch_tokens = max(1024, budget)  # sticky: GPU is shared, stay conservative
                print(f"  [rerank] OOM -> batch_tokens={budget}", flush=True)
                time.sleep(1)
        return out

    def score(self, query: str, texts: list[str]) -> np.ndarray:
        """Raw logits for (query, text_i), using the sqlite cache."""
        if not self.use_cache:
            return self.logits([(query, t) for t in texts])
        db = self._conn()
        keys = [self._key(query, t) for t in texts]
        found = {}
        uk = list(dict.fromkeys(keys))
        for s in range(0, len(uk), 900):
            part = uk[s:s + 900]
            for k, v in db.execute(f"SELECT k, v FROM rr WHERE k IN ({','.join('?' * len(part))})", part):
                found[k] = v
        miss = {}
        for k, t in zip(keys, texts):
            if k not in found:
                miss.setdefault(k, t)
        if miss:
            mk = list(miss)
            lg = self.logits([(query, miss[k]) for k in mk])
            for k, v in zip(mk, lg):
                found[k] = float(v)
            try:
                db.executemany("INSERT OR REPLACE INTO rr VALUES (?, ?)", [(k, float(v)) for k, v in zip(mk, lg)])
                db.commit()
            except sqlite3.OperationalError as e:
                print(f"  [rerank] cache write skipped: {e}")
        return np.array([found[k] for k in keys], dtype=np.float32)

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None


_RR: dict[str, Reranker] = {}


def get_reranker(cfg: dict | None = None) -> Reranker:
    r = rrcfg(cfg)
    key = f"{r['model']}|{r['max_length']}"
    if key not in _RR:
        _RR[key] = Reranker(r["model"], r["max_length"], r["batch_tokens"], r["fp16"], r.get("device"),
                            r["cache"], r["cache_path"])
    return _RR[key]


def release_reranker():
    import torch
    for m in _RR.values():
        m.close()
    _RR.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def rerank(query_vi: str, candidates: list[dict], top_n: int | None = None, cfg: dict | None = None) -> list[dict]:
    """Rerank candidates by cross-encoder; returns new dicts sorted by calibrated score desc."""
    if not candidates:
        return []
    r = rrcfg(cfg)
    top_n = top_n if top_n is not None else r.get("top_n")
    rr = get_reranker(cfg)
    lg = rr.score(query_vi, [c["text"] for c in candidates])
    prob = _sigmoid(lg)
    out = []
    for c, l, p in zip(candidates, lg, prob):
        d = dict(c)
        d["retrieval_score"] = c.get("score")
        d["logit"] = float(l)
        d["score"] = float(p)
        d["stage"] = "rerank"
        out.append(d)
    out.sort(key=lambda x: -x["score"])
    return out[:top_n] if top_n else out
