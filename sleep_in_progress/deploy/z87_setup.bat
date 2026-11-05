@echo off
rem Cai Python portable (uv) + torch CUDA + bge-m3 tren z87. Chi tai GPU; chay uu tien thap.
rem Log: D:\r2ai\work\setup.log
set PYTHONUNBUFFERED=1
set UV_CONCURRENT_DOWNLOADS=1
set UV_PYTHON_INSTALL_DIR=D:\r2ai\py
set UV_CACHE_DIR=D:\r2ai\uvcache
set HF_HOME=D:\r2ai\hf
cd /d D:\r2ai
echo [%date% %time%] start > work\setup.log
if not exist tools mkdir tools
if not exist tools\uv.exe (
  curl.exe -sL --limit-rate 4m -o tools\uv.zip https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip >> work\setup.log 2>&1
  powershell -NoProfile -Command "Expand-Archive -Force D:\r2ai\tools\uv.zip D:\r2ai\tools" >> work\setup.log 2>&1
)
echo [%date% %time%] uv ready >> work\setup.log
tools\uv.exe --version >> work\setup.log 2>&1
tools\uv.exe venv --python 3.11 D:\r2ai\venv >> work\setup.log 2>&1
echo [%date% %time%] venv done >> work\setup.log
tools\uv.exe pip install --python D:\r2ai\venv\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 >> work\setup.log 2>&1
echo [%date% %time%] torch done >> work\setup.log
tools\uv.exe pip install --python D:\r2ai\venv\Scripts\python.exe numpy==1.26.4 transformers==4.49.0 huggingface_hub==0.29.3 zstandard==0.23.0 >> work\setup.log 2>&1
echo [%date% %time%] libs done >> work\setup.log
D:\r2ai\venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download as s; p=s('BAAI/bge-m3', allow_patterns=['*.json','pytorch_model.bin','sentencepiece.bpe.model','tokenizer.json'], max_workers=1); print(p)" >> work\setup.log 2>&1
echo [%date% %time%] model done >> work\setup.log
D:\r2ai\venv\Scripts\python.exe -c "import torch;print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))" >> work\setup.log 2>&1
echo [%date% %time%] DONE >> work\setup.log
