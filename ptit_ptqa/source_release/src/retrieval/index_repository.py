"""Index repository for discovering, filtering, and loading index shards."""

import glob
import logging
import os
from typing import Dict, Iterator, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)
VOCAB_SIZE = 250005


class IndexShard:
    """Represents a discrete index shard containing chunks, texts, and embeddings."""

    def __init__(self, shard_dir: str):
        self.shard_dir = shard_dir
        self.chunks_dir = os.path.join(shard_dir, "chunks")
        self.text_dir = os.path.join(shard_dir, "text")
        self.dense_dir = os.path.join(shard_dir, "emb_dense")

    @property
    def name(self) -> str:
        return os.path.basename(self.shard_dir)

    def load_chunks(
        self,
        valid_doc_ids: Optional[Set[int]] = None
    ) -> Tuple[Dict[int, Tuple[int, int, int]], int]:
        """
        Loads mapping: chunk_id -> (doc_id, start, end).
        Returns (chunks_map, total_chunks_before_filter).
        """
        chunks_map: Dict[int, Tuple[int, int, int]] = {}
        total_chunks = 0

        p_files = glob.glob(os.path.join(self.chunks_dir, "part_*.parquet"))
        for p_file in p_files:
            c_df = pd.read_parquet(p_file, columns=["chunk_id", "doc_id", "start", "end"])
            total_chunks += len(c_df)
            if valid_doc_ids is not None:
                c_df = c_df[c_df["doc_id"].isin(valid_doc_ids)]
            for row in c_df.itertuples(index=False):
                chunks_map[row.chunk_id] = (int(row.doc_id), int(row.start), int(row.end))

        return chunks_map, total_chunks

    def load_texts(
        self,
        valid_doc_ids: Optional[Set[int]] = None
    ) -> Dict[int, str]:
        """Loads mapping: doc_id -> text."""
        texts_map: Dict[int, str] = {}
        p_files = glob.glob(os.path.join(self.text_dir, "part_*.parquet"))
        for p_file in p_files:
            t_df = pd.read_parquet(p_file, columns=["doc_id", "text"])
            if valid_doc_ids is not None:
                t_df = t_df[t_df["doc_id"].isin(valid_doc_ids)]
            for row in t_df.itertuples(index=False):
                texts_map[int(row.doc_id)] = str(row.text)

        return texts_map

    def iter_embedding_blocks(
        self,
        need_dense: bool = True,
        need_sparse: bool = True
    ) -> Iterator[Tuple[np.ndarray, Optional[np.ndarray], Optional[sp.csr_matrix]]]:
        """
        Yields (chunk_ids, dense_vectors, sparse_csr_matrix) per block.
        """
        id_files = sorted(glob.glob(os.path.join(self.dense_dir, "ids_*.npy")))
        for id_file in id_files:
            c_ids = np.load(id_file)
            num_chunks = len(c_ids)

            c_vecs: Optional[np.ndarray] = None
            if need_dense:
                vec_file = id_file.replace("ids_", "vec_")
                c_vecs = np.load(vec_file).astype(np.float32)

            M: Optional[sp.csr_matrix] = None
            if need_sparse:
                sp_file = id_file.replace("ids_", "sp_").replace(".npy", ".npz")
                if os.path.exists(sp_file):
                    z = np.load(sp_file)
                    M = sp.csr_matrix(
                        (z["vals"].astype(np.float32), z["ids"], z["ptr"]),
                        shape=(num_chunks, VOCAB_SIZE)
                    )

            yield c_ids, c_vecs, M


class IndexRepository:
    """Manages index shard discovery and valid doc IDs loading."""

    def __init__(self, index_base: str):
        self.index_base = index_base

    def discover_shards(self) -> List[IndexShard]:
        """
        Scans base directory for index shards with automatic signature-based deduplication.
        """
        chunk_marker_dirs = sorted(
            glob.glob(os.path.join(self.index_base, "**", "chunks"), recursive=True)
        )
        unique_shards: List[IndexShard] = []
        seen_signatures = {}

        logger.info(f"Scanning for index shards in {self.index_base}...")
        for marker_dir in chunk_marker_dirs:
            s_dir = os.path.dirname(marker_dir)
            p_files = sorted(glob.glob(os.path.join(marker_dir, "part_*.parquet")))
            if not p_files:
                continue

            # Read first 5 doc IDs of part 0 to construct a unique fingerprint
            df_sig = pd.read_parquet(p_files[0], columns=["doc_id"])
            sig = (len(p_files), len(df_sig), tuple(df_sig["doc_id"].iloc[:5].tolist()))
            rel_path = os.path.relpath(s_dir, self.index_base).replace("\\", "/")

            if sig in seen_signatures:
                prev_rel = os.path.relpath(seen_signatures[sig], self.index_base).replace("\\", "/")
                logger.warning(
                    f"[DEDUPLICATE SKIP] Shard '{rel_path}' is duplicate of '{prev_rel}' -> Skipping."
                )
            else:
                seen_signatures[sig] = s_dir
                unique_shards.append(IndexShard(s_dir))

        return unique_shards

    @staticmethod
    def load_valid_doc_ids(btc_links_file: Optional[str]) -> Optional[Set[int]]:
        """Loads valid document ID whitelist from BTC links parquet."""
        if not btc_links_file or not os.path.exists(btc_links_file):
            return None

        logger.info(f"Loading valid document IDs from {btc_links_file}...")
        btc_links = pd.read_parquet(btc_links_file, columns=["id"])
        valid_ids = set(btc_links["id"].astype(int))
        logger.info(f"Loaded {len(valid_ids):,} valid document IDs.")
        return valid_ids
