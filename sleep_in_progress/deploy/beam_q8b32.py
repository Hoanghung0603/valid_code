"""Beam Pod for q8b32: Qwen3-Reranker-8B on RTX4090."""
from beam import Image, Pod, Volume

data_volume = Volume(name="r2ai-data", mount_path="/data")
hf_volume = Volume(name="r2ai-hf-cache", mount_path="/hf-cache")

image = Image(
    python_version="python3.11",
    commands=[
        "pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124",
        "pip install transformers==4.51.3 zstandard==0.23.0 numpy==1.26.4 huggingface_hub==0.30.2",
    ],
)

pod = Pod(
    name="q8b32-rerank",
    image=image,
    volumes=[data_volume, hf_volume],
    gpu="RTX4090",
    cpu=4,
    memory="32Gi",
    entrypoint=["python", "/mnt/code/deploy/run_q8b32.py"],
    keep_warm_seconds=10800,
)
