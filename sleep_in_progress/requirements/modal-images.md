# Môi trường Modal trong chuỗi vi53/vi56

Thông số lấy từ định nghĩa image trong mã hiện tại ngày 2026-11-04, không phải lockfile lúc build; CUDA/driver và dependency bắc cầu chưa được ghim đầy đủ. Nên đọc từng `deploy/modal_*.py` để biết chi tiết và kiểm revision model.

- `modal_runner.py`, `modal_vi41_score.py`: Debian Python 3.11, torch 2.6.0, transformers 4.51.3, numpy 1.26.4, zstandard 0.23.0.
- `modal_vi41.py`, `modal_vi45.py`, `modal_vi50.py`: vLLM 0.15.1, zstandard 0.23.0.
- `modal_vi51.py`, `modal_vi55.py`: torch 2.9.1, transformers 4.57.6, zstandard 0.23.0.
- `modal_vi56.py`: Qwen vLLM 0.15.1; BGE torch 2.6.0, transformers 4.51.3, numpy 1.26.4, huggingface_hub 0.30.2.
- `modal_vi41_search.py`: Debian Python 3.11, torch 2.6.0, transformers 4.51.3, numpy 1.26.4, huggingface_hub 0.30.2, safetensors 0.5.3, sentencepiece 0.2.0.
- `modal_q8b36.py`: Debian Python 3.11, torch 2.6.0, transformers 4.51.3, zstandard 0.23.0, numpy 1.26.4, huggingface_hub 0.30.2.
- GPU z87 và LM Studio: môi trường thực tế/chính xác chưa được khóa; các snapshot CPU không đại diện.
