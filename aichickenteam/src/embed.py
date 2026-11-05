"""Dense text encoder (BGE-M3 dense by default; Qwen3-Embedding supported).

- fp16 on CUDA, length-sorted token-budget batching (little padding, bounded VRAM)
- automatic OOM back-off (halves the token budget and retries)
- optional sqlite cache keyed by sha1(model|text) for on-the-fly chunks
  (PubMed candidates, expanded doc chunks) -> data/cache/embed.sqlite

Usage:
    enc = get_encoder(cfg)               # singleton per model
    vecs = enc.encode(texts)             # np.float16 [n, d], L2-normalised
    qv = enc.encode(queries, is_query=True)
    release_encoder()                    # free VRAM
"""
from __future__ import annotations

import gc
import hashlib
import os
import sqlite3
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# Local fp16 safetensors conversion of BAAI/bge-m3 (the hub repo only ships
# pytorch_model.bin, which transformers>=5 refuses to load with torch<2.6).
LOCAL_MODELS = {"BAAI/bge-m3": ROOT / "data/models/bge-m3"}

QWEN3_QUERY_INSTRUCT = ("Instruct: Given a medical question, retrieve relevant passages "
                        "that answer the question\nQuery: ")


def _resolve(name: str) -> str:
    p = LOCAL_MODELS.get(name)
    if p is not None and (p / "model.safetensors").exists():
        return str(p)
    return name


def dense_cfg(cfg: dict | None) -> dict:
    cfg = cfg or {}
    d = dict(((cfg.get("index") or {}).get("dense") or {}))
    d.setdefault("model", "BAAI/bge-m3")
    d.setdefault("max_length", 512)
    d.setdefault("query_max_length", 256)
    d.setdefault("batch_tokens", 16384)
    d.setdefault("fp16", True)
    return d


class Encoder:
    def __init__(self, model: str = "BAAI/bge-m3", max_length: int = 512,
                 batch_tokens: int = 16384, fp16: bool = True, device: str | None = None,
                 pooling: str | None = None, query_max_length: int = 256, cache_path: str | None = None):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.name = model
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        path = _resolve(model)
        self.is_qwen = "qwen" in model.lower()
        self.pooling = pooling or ("last" if self.is_qwen else "cls")
        self.tok = AutoTokenizer.from_pretrained(path, padding_side="left" if self.pooling == "last" else "right")
        dtype = torch.float16 if (fp16 and self.device == "cuda") else torch.float32
        self.model = AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.max_length = max_length
        self.query_max_length = query_max_length
        self.batch_tokens = batch_tokens
        self.dim = int(self.model.config.hidden_size)
        self._cache_path = cache_path
        self._db = None

    # ------------------------------------------------------------------ core
    def _batches(self, lengths: np.ndarray, budget: int, max_bs: int = 256):
        order = np.argsort(-lengths, kind="stable")
        i = 0
        while i < len(order):
            L = int(lengths[order[i]])
            bs = max(1, min(max_bs, budget // max(L, 1)))
            yield order[i:i + bs]
            i += bs

    def encode(self, texts: list[str], is_query: bool = False, max_length: int | None = None,
               show_progress: bool = False) -> np.ndarray:
        torch = self.torch
        if len(texts) == 0:
            return np.zeros((0, self.dim), dtype=np.float16)
        if is_query and self.is_qwen:
            texts = [QWEN3_QUERY_INSTRUCT + t for t in texts]
        ml = max_length or (self.query_max_length if is_query else self.max_length)
        enc = self.tok(list(texts), truncation=True, max_length=ml, padding=False)
        ids = enc["input_ids"]
        lengths = np.array([len(x) for x in ids])
        out = np.zeros((len(texts), self.dim), dtype=np.float16)
        budget = self.batch_tokens
        batches = list(self._batches(lengths, budget))
        bi = 0
        t0 = time.time()
        done = 0
        while bi < len(batches):
            idx = batches[bi]
            feats = None
            try:
                feats = self.tok.pad({"input_ids": [ids[j] for j in idx]}, return_tensors="pt")
                feats = {k: v.to(self.device) for k, v in feats.items()}
                with torch.inference_mode():
                    h = self.model(**feats).last_hidden_state
                    if self.pooling == "cls":
                        v = h[:, 0]
                    elif self.pooling == "last":
                        v = h[:, -1]  # left padding
                    else:  # mean
                        m = feats["attention_mask"].unsqueeze(-1).to(h.dtype)
                        v = (h * m).sum(1) / m.sum(1).clamp(min=1)
                    v = torch.nn.functional.normalize(v.float(), dim=-1)
                out[idx] = v.cpu().numpy().astype(np.float16)
                done += len(idx)
                bi += 1
                if show_progress and bi % 50 == 0:
                    el = time.time() - t0
                    print(f"  embed {done}/{len(texts)} {done / max(el, 1e-6):.1f}/s", flush=True)
            except torch.cuda.OutOfMemoryError:
                feats = None
                gc.collect()
                torch.cuda.empty_cache()
                if budget <= 512 and len(idx) == 1:
                    raise
                budget = max(512, budget // 2)
                self.batch_tokens = max(1024, budget)  # sticky: GPU is shared
                print(f"  [embed] OOM -> batch_tokens={budget}", flush=True)
                # re-split the remaining work with the smaller budget
                rest = np.concatenate([batches[j] for j in range(bi, len(batches))])
                sub = lengths[rest]
                batches = batches[:bi] + [rest[b] for b in self._batches(sub, budget)]
                time.sleep(1)
        return out

    # ----------------------------------------------------------------- cache
    def _conn(self):
        if self._db is None:
            p = Path(self._cache_path or "data/cache/embed.sqlite")
            p = p if p.is_absolute() else ROOT / p
            p.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(str(p), timeout=60)
            self._db.execute("CREATE TABLE IF NOT EXISTS emb (k TEXT PRIMARY KEY, v BLOB)")
        return self._db

    def _key(self, t: str) -> str:
        return hashlib.sha1(f"{self.name}|{self.max_length}|{t}".encode()).hexdigest()

    def encode_cached(self, texts: list[str]) -> np.ndarray:
        """Passage encoding with persistent sqlite cache (for on-the-fly chunks)."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float16)
        db = self._conn()
        keys = [self._key(t) for t in texts]
        found: dict[str, np.ndarray] = {}
        uniq = list(dict.fromkeys(keys))
        for i in range(0, len(uniq), 900):
            part = uniq[i:i + 900]
            q = f"SELECT k, v FROM emb WHERE k IN ({','.join('?' * len(part))})"
            for k, v in db.execute(q, part):
                found[k] = np.frombuffer(v, dtype=np.float16)
        miss_idx = [i for i, k in enumerate(keys) if k not in found]
        if miss_idx:
            # dedupe misses
            mk = {}
            for i in miss_idx:
                mk.setdefault(keys[i], texts[i])
            mkeys = list(mk)
            vecs = self.encode([mk[k] for k in mkeys])
            for k, v in zip(mkeys, vecs):
                found[k] = v
            try:
                db.executemany("INSERT OR REPLACE INTO emb VALUES (?, ?)",
                               [(k, v.tobytes()) for k, v in zip(mkeys, vecs)])
                db.commit()
            except sqlite3.OperationalError as e:  # concurrent writer; cache is best-effort
                print(f"  [embed] cache write skipped: {e}")
        return np.stack([found[k] for k in keys]).astype(np.float16)

    def close(self):
        if self._db is not None:
            self._db.close()
            self._db = None


_ENCODERS: dict[str, Encoder] = {}


def get_encoder(cfg: dict | None = None) -> Encoder:
    d = dense_cfg(cfg)
    key = d["model"]
    if key not in _ENCODERS:
        _ENCODERS[key] = Encoder(model=d["model"], max_length=d["max_length"], batch_tokens=d["batch_tokens"],
                                 fp16=d["fp16"], query_max_length=d["query_max_length"],
                                 device=d.get("device"), cache_path=d.get("cache_path"))
    return _ENCODERS[key]


def release_encoder():
    import torch
    for e in _ENCODERS.values():
        e.close()
    _ENCODERS.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
