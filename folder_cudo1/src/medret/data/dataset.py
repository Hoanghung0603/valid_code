from __future__ import annotations

from pathlib import Path

import pandas as pd

from medret.config import Config, get_config


def download_dataset(cfg: Config | None = None) -> Path:
    """Snapshot the HF dataset repo into raw_dir. Set HF_TOKEN if the repo is gated."""
    from huggingface_hub import snapshot_download

    cfg = cfg or get_config()
    target = cfg.paths.raw_dir / cfg.dataset.hf_repo.split("/")[-1]
    snapshot_download(repo_id=cfg.dataset.hf_repo, repo_type="dataset", local_dir=target)
    return target


def _find(cfg: Config, filename: str) -> Path:
    # The repo layout may nest files (e.g. under data/), so search instead of hard-coding.
    matches = sorted(cfg.paths.raw_dir.rglob(filename))
    if not matches:
        raise FileNotFoundError(f"{filename} not found under {cfg.paths.raw_dir}; run `medret download`")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple {filename} under {cfg.paths.raw_dir}: {matches}")
    return matches[0]


def load_queries(cfg: Config | None = None) -> pd.DataFrame:
    """Columns: id (int), query (str, Vietnamese)."""
    cfg = cfg or get_config()
    return pd.read_parquet(_find(cfg, cfg.dataset.queries_file))


def load_corpus(cfg: Config | None = None) -> pd.DataFrame:
    """Columns: id (int), url (str). Millions of rows."""
    cfg = cfg or get_config()
    return pd.read_parquet(_find(cfg, cfg.dataset.corpus_file))
