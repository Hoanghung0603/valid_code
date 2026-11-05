"""Tải BAAI/bge-reranker-v2-m3 vào HF cache: chỉ file cần thiết (safetensors, không tải pytorch_model.bin)."""
import sys

sys.stdout.reconfigure(encoding="utf-8")

from huggingface_hub import snapshot_download

p = snapshot_download(
    "BAAI/bge-reranker-v2-m3",
    allow_patterns=[
        "config.json",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    ],
    max_workers=1,
)
print(p, flush=True)
