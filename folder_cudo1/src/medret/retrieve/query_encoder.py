"""BGE-M3 query encoder on CPU, matching the passage side of the Kaggle runner.

Same model, `query_input` text (no instruction), [CLS] of the last hidden state, L2-normalised (paper
2402.03216; FlagEmbedding recipe). Passages were encoded in fp16 on T4s; CPU fp32 query vectors are
compatible (fp16 vs fp32 cosine >= 0.99996 in docs/experiments/20261010-embedding-throughput-kaggle-t4.md).
Vectors are cached by sha1(model, input text), so re-running with edited query variants only encodes the new ones.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from medret.cache import KVCache, cache_key
from medret.index.embed_input import query_input


class QueryEncoder:
    def __init__(self, model: str = "BAAI/bge-m3", cache_path: Path | None = None, max_length: int = 512,
                 batch_size: int = 32, threads: int | None = None, dtype: str = "float32"):
        self.model_name, self.max_length, self.batch_size = model, max_length, batch_size
        # bfloat16 halves the CPU model RAM (~1.15 GB vs ~2.3 GB) but is emulated (very slow) on CPUs without AVX512-BF16;
        # its vectors are cached under their own key, never mixed with the float32 ones
        self.dtype = dtype
        self._key_model = model if dtype == "float32" else f"{model}:{dtype}"
        self.cache = KVCache(cache_path) if cache_path else None
        self._model = self._tok = None
        self.threads = threads

    def _load(self) -> None:
        if self._model is None:
            import torch
            from transformers import AutoModel, AutoTokenizer

            if self.threads:
                torch.set_num_threads(self.threads)
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModel.from_pretrained(self.model_name, dtype=getattr(torch, self.dtype)).eval()

    def _encode(self, texts: list[str]) -> np.ndarray:
        import torch

        self._load()
        out = []
        order = np.argsort([len(t) for t in texts])  # length-sorted batches: little padding
        with torch.inference_mode():
            for s in range(0, len(texts), self.batch_size):
                idx = order[s : s + self.batch_size]
                enc = self._tok([texts[i] for i in idx], padding=True, truncation=True, max_length=self.max_length,
                                return_tensors="pt")
                cls = self._model(**enc).last_hidden_state[:, 0]
                out.append((idx, torch.nn.functional.normalize(cls.float(), dim=-1).numpy()))
        vecs = np.empty((len(texts), out[0][1].shape[1]), dtype=np.float32)
        for idx, v in out:
            vecs[idx] = v
        return vecs

    def encode(self, queries: list[str]) -> np.ndarray:
        """L2-normalised float32 vectors, one row per query (cached)."""
        texts = [query_input(q) for q in queries]
        keys = [cache_key("qvec", self._key_model, t) for t in texts]
        hit = self.cache.get_many(keys) if self.cache else {}
        miss = list(dict.fromkeys(i for i, k in enumerate(keys) if k not in hit))
        new: dict[str, np.ndarray] = {}
        for s in range(0, len(miss), 512):  # cached block by block: an interrupted run keeps what it encoded
            part = miss[s : s + 512]
            vecs = self._encode([texts[i] for i in part])
            for j, i in enumerate(part):
                new[keys[i]] = vecs[j]
            if self.cache:
                self.cache.put_many((keys[i], vecs[j].tobytes()) for j, i in enumerate(part))
        return np.stack([np.frombuffer(hit[k], dtype=np.float32) if k in hit else new[k] for k in keys])
