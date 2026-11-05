"""Multi-channel candidate retrieval + reciprocal rank fusion (architecture.md section 2.5).

Channels per query (each optional, skipped when its index or input is missing):
  bm25_vi   Vietnamese query, pruned terms, on the vi chunk index
  bm25_zh   Chinese translation, pruned terms, on the zh chunk index
  dense_vi  BGE-M3 vector of the Vietnamese query (cross-lingual: finds zh chunks too)
  dense_zh  BGE-M3 vector of the Chinese translation
  bm25_en   English translation, pruned terms, on the en (PubMed) chunk index
  dense_en  BGE-M3 vector of the English translation
Every channel keeps only chunks of docs in the corpus scope (index/scope.py) when a scope file exists.
Dense hits are inputs (deduplicated texts); each input expands to every chunk carrying it.
RRF: score(chunk) = sum over channels of 1 / (rrf_k + rank). Scores of different channels are never compared.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

STOP_VI = frozenset(
    "em anh chị bác sĩ tôi mình con cháu ạ ơi dạ vâng cho hỏi xin cảm ơn nhé thì là mà và của có không được bị "
    "với các những một này đó kia như thế nào gì sao vậy hay hoặc nhưng vì nếu để khi đã đang sẽ rất cũng nên "
    "phải cần làm ra vào lên xuống thấy nói biết muốn".split()
)
STOP_EN = frozenset(
    "the a an of and or in on for to with by is are was were be been being as at from that this these those it its "
    "what which who how can could should would will do does did my me i you your patient doctor".split()
)
STOP_ZH = frozenset(
    "请问 医生 您好 你好 谢谢 我的 是否 是不是 有没有 怎么 怎么办 什么 需要 可以 应该 一下 这个 那个 这样".split()
)


@dataclass
class RetrieveConfig:
    k_per_channel: int = 200
    rrf_k: int = 60
    n_candidates: int = 300
    bm25_max_df: float = 0.1
    channels: list[str] = field(default_factory=lambda: ["bm25_vi", "bm25_zh", "dense_vi", "dense_zh"])


class Retriever:
    def __init__(self, index_dir: Path, dense_run: str, nprobe: int = 64, k_coarse: int = 1000,
                 cfg: RetrieveConfig | None = None, log=print):
        from medret.index.lexical import BM25Index
        from medret.index.meta import DB_FILE, MetaStore

        self.cfg = cfg or RetrieveConfig()
        self.meta = MetaStore(index_dir / DB_FILE)
        self.bm25 = {lang: BM25Index(index_dir / "bm25" / lang) for lang in ("vi", "zh", "en")
                     if (index_dir / "bm25" / lang).exists()}
        from medret.index.scope import load_scope

        scope = load_scope(index_dir)
        self.scope = frozenset(scope[0].tolist()) if scope is not None else None
        log(f"scope: {'none' if self.scope is None else f'{len(self.scope):,} docs'}")
        self.dense = None
        dense_dir = index_dir / "dense" / dense_run
        if (dense_dir / "ivfpq.faiss").exists():
            from medret.index.dense import DenseIndex

            self.dense = DenseIndex(dense_dir, nprobe=nprobe)
            self.k_coarse = k_coarse
            log(f"dense: {self.dense.meta['n_present']:,}/{self.dense.meta['n']:,} vectors indexed")
        else:
            log(f"dense: no index at {dense_dir}, dense channels skipped")
        log(f"bm25: {sorted(self.bm25)}")

    def _dense_chunks(self, qvec: np.ndarray, k: int) -> list[int]:
        hits = self.dense.search(qvec[None, :], k=k, k_coarse=max(self.k_coarse, k))[0]
        rank_of = {iid: r for r, (iid, _) in enumerate(hits)}
        rows = self.meta.chunks_of_inputs(list(rank_of))
        if rows.empty:
            return []
        rows = rows.assign(r=rows["input_id"].map(rank_of)).sort_values(["r", "chunk_id"])
        return rows["chunk_id"].tolist()

    def retrieve(self, q_vi: str, q_zh: str | None, v_vi: np.ndarray | None, v_zh: np.ndarray | None,
                 q_en: str | None = None, v_en: np.ndarray | None = None) -> pd.DataFrame:
        """Candidates of one query: chunk_id, rrf, routes ("bm25_vi:3|dense_vi:0"), best_rank."""
        from medret.chunk.run import CHUNK_ID_STRIDE

        c, k = self.cfg, self.cfg.k_per_channel
        ranked: dict[str, list[int]] = {}
        if "bm25_vi" in c.channels and "vi" in self.bm25:
            ranked["bm25_vi"] = [i for i, _ in self.bm25["vi"].search(q_vi, k, c.bm25_max_df, STOP_VI)]
        if "bm25_zh" in c.channels and "zh" in self.bm25 and q_zh:
            ranked["bm25_zh"] = [i for i, _ in self.bm25["zh"].search(q_zh, k, c.bm25_max_df, STOP_ZH)]
        if self.dense is not None:
            if "dense_vi" in c.channels and v_vi is not None:
                ranked["dense_vi"] = self._dense_chunks(v_vi, k)
            if "dense_zh" in c.channels and v_zh is not None:
                ranked["dense_zh"] = self._dense_chunks(v_zh, k)
            if "dense_en" in c.channels and v_en is not None:
                ranked["dense_en"] = self._dense_chunks(v_en, k)
        if "bm25_en" in c.channels and "en" in self.bm25 and q_en:
            ranked["bm25_en"] = [i for i, _ in self.bm25["en"].search(q_en, k, c.bm25_max_df, STOP_EN)]
        if self.scope is not None:  # ranks are re-counted after the filter
            ranked = {ch: [i for i in ids if i // CHUNK_ID_STRIDE in self.scope] for ch, ids in ranked.items()}
        scores: dict[int, float] = {}
        routes: dict[int, list[str]] = {}
        best: dict[int, int] = {}
        for ch, ids in ranked.items():
            for r, cid in enumerate(ids):
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (c.rrf_k + r + 1)
                routes.setdefault(cid, []).append(f"{ch}:{r}")
                best[cid] = min(best.get(cid, r), r)
        if not scores:
            return pd.DataFrame(columns=["chunk_id", "rrf", "routes", "best_rank"])
        df = pd.DataFrame({"chunk_id": list(scores), "rrf": list(scores.values())})
        df["routes"] = df["chunk_id"].map(lambda i: "|".join(routes[i]))
        df["best_rank"] = df["chunk_id"].map(best)
        return df.sort_values("rrf", ascending=False).head(c.n_candidates).reset_index(drop=True)
