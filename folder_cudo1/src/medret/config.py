"""Typed project config loaded from YAML (configs/default.yaml unless MEDRET_CONFIG is set)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"


class PathsConfig(BaseModel):
    data_dir: Path
    raw_dir: Path
    crawl_dir: Path
    parsed_dir: Path
    chunks_dir: Path
    index_dir: Path
    dev_dir: Path
    runs_dir: Path
    submissions_dir: Path


class DatasetConfig(BaseModel):
    hf_repo: str
    queries_file: str
    corpus_file: str


class EvalConfig(BaseModel):
    tokenizer: str = "whitespace"  # spec of 2026-10-10 names no tokenizer
    relevance_threshold: float = 0.4
    near_duplicate_threshold: float | None = None  # earlier spec only
    beta: float = 2.0


class EmbedConfig(BaseModel):
    model: str = "BAAI/bge-m3"  # also the tokenizer that sizes chunks (medret.chunk.packer)
    max_length: int = 1024  # chunks are <= ~512 tokens plus a title / heading prefix


class IndexConfig(BaseModel):
    dense_run: str = "bge-m3-v1"  # data/embeddings/<run> (runner output) -> <index_dir>/dense/<run>
    faiss_factory: str = "IVF8192,PQ64"  # inner product; ~70 B per vector in RAM
    nprobe: int = 64
    k_coarse: int = 1000  # IVF-PQ candidates re-scored exactly from the fp16 memmap
    bm25_workers: int = 6
    meta_memory_limit: str = "6GB"


class LLMConfig(BaseModel):
    # OpenAI-compatible endpoint (FPT AI Marketplace, FPT_API_BASE / FPT_API_KEY in .env)
    translate_model: str = "Qwen3.6-27B"  # DEV ONLY (27B > 15B): swap for a <= 15B model before any submission
    threads: int = 8


class SubmissionConfig(BaseModel):
    doc_id_type: Literal["int", "str"] = "int"


class Config(BaseModel):
    paths: PathsConfig
    dataset: DatasetConfig
    eval: EvalConfig = EvalConfig()
    embed: EmbedConfig = EmbedConfig()
    index: IndexConfig = IndexConfig()
    llm: LLMConfig = LLMConfig()
    submission: SubmissionConfig = SubmissionConfig()


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path or os.environ.get("MEDRET_CONFIG") or DEFAULT_CONFIG)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    cfg = Config.model_validate(raw)
    # Relative paths are relative to the project root, not the cwd.
    for name, value in cfg.paths:
        if not value.is_absolute():
            setattr(cfg.paths, name, PROJECT_ROOT / value)
    return cfg


@lru_cache(maxsize=1)
def get_config() -> Config:
    return load_config()
