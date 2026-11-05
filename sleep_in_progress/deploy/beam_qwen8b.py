"""Beam Pod: Qwen3-Reranker-8B rerank (CausalLM yes/no, sdpa) trên RTX4090.

Chạy qua: beam run deploy/beam_qwen8b.py:pod --detach --env CANDS_FILE=... --env ...
Env: CANDS_FILE, OUT_FILE, RESUME_DIR, MAX_LEN, BATCH, DTYPE, LIMIT_QUERIES
"""
import os
import subprocess
import sys
import time

from beam import Image, Pod, Volume

DATA_VOL = "/data"
HF_CACHE = "/hf-cache"

data_volume = Volume(name="r2ai-data", mount_path=DATA_VOL)
hf_volume = Volume(name="r2ai-hf-cache", mount_path=HF_CACHE)

image = Image(
    python_version="python3.11",
    commands=[
        "pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
        "pip install transformers==4.51.3 zstandard==0.23.0 numpy==1.26.4 huggingface_hub==0.30.2",
    ],
)

pod = Pod(
    name="qwen8b-rerank",
    image=image,
    volumes=[data_volume, hf_volume],
    gpu="RTX4090",
    cpu=4,
    memory="32Gi",
    entrypoint=["python", "/mnt/code/deploy/run_shard.py"],
    keep_warm_seconds=21600,
)
