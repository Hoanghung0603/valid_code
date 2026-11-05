"""Query encoder implementation using BGE-M3 model."""

import gc
import logging
from typing import Dict, List, Optional
import numpy as np
import scipy.sparse as sp

from src.retrieval.base import BaseQueryEncoder

logger = logging.getLogger(__name__)
DEFAULT_VOCAB_SIZE = 250005


class BGEM3QueryEncoder(BaseQueryEncoder):
    """Encapsulates BAAI/bge-m3 for dense vector and sparse lexical query encoding."""

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        vocab_size: int = DEFAULT_VOCAB_SIZE
    ):
        self.model_name = model_name
        self.use_fp16 = use_fp16
        self.vocab_size = vocab_size
        self._model = None

    def _ensure_model_loaded(self) -> None:
        if self._model is None:
            logger.info(f"Loading {self.model_name} (use_fp16={self.use_fp16})...")
            from FlagEmbedding import BGEM3FlagModel
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self.use_fp16)

    def encode_dense(
        self,
        queries: List[str],
        batch_size: int = 8,
        max_length: int = 1024
    ) -> np.ndarray:
        self._ensure_model_loaded()
        enc = self._model.encode(
            queries,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return np.array(enc["dense_vecs"], dtype=np.float32)

    def build_query_sparse_matrix(
        self,
        lexical_weights_list: List[Dict[str, float]]
    ) -> sp.csr_matrix:
        rows, cols, data = [], [], []
        for q_idx, weights in enumerate(lexical_weights_list):
            for tok_id_str, w in weights.items():
                tok_id = int(tok_id_str)
                if tok_id < self.vocab_size:
                    rows.append(q_idx)
                    cols.append(tok_id)
                    data.append(float(w))

        if not data:
            return sp.csr_matrix(
                (len(lexical_weights_list), self.vocab_size),
                dtype=np.float32
            )
        return sp.csr_matrix(
            (data, (rows, cols)),
            shape=(len(lexical_weights_list), self.vocab_size),
            dtype=np.float32,
        )

    def encode_sparse(
        self,
        queries_vi: List[str],
        queries_zh: Optional[List[str]] = None,
        queries_en: Optional[List[str]] = None,
        zh_weight: float = 1.0,
        en_weight: float = 0.5,
        batch_size: int = 8,
        max_length: int = 1024
    ) -> sp.csr_matrix:
        self._ensure_model_loaded()
        enc_vi = self._model.encode(
            queries_vi,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=False,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        fused_lexical = [dict(w) for w in enc_vi["lexical_weights"]]

        # Merge Chinese lexical weights
        if queries_zh is not None and any(bool(q) for q in queries_zh):
            enc_zh = self._model.encode(
                queries_zh,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            for merged, w_z in zip(fused_lexical, enc_zh["lexical_weights"]):
                for tok, val in w_z.items():
                    merged[tok] = max(merged.get(tok, 0.0), float(val) * zh_weight)

        # Merge English lexical weights
        if queries_en is not None and any(bool(q) for q in queries_en):
            enc_en = self._model.encode(
                queries_en,
                batch_size=batch_size,
                max_length=max_length,
                return_dense=False,
                return_sparse=True,
                return_colbert_vecs=False,
            )
            for merged, w_e in zip(fused_lexical, enc_en["lexical_weights"]):
                for tok, val in w_e.items():
                    merged[tok] = max(merged.get(tok, 0.0), float(val) * en_weight)

        return self.build_query_sparse_matrix(fused_lexical)

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Unloaded BGEM3QueryEncoder and cleared GPU VRAM.")
