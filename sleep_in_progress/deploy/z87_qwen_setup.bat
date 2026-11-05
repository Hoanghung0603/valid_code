@echo off
rem Cai venv_qwen (uv, Python D:\r2ai\py) + torch 2.6.0 cu124 + transformers 4.51.3 + Qwen3-Reranker-0.6B tren z87.
rem Khong dong D:\r2ai\venv cu. Chi tai GPU; chay uu tien thap. Log: D:\r2ai\work\qrr06\setup.log (cuoi: exit done)
set PYTHONUNBUFFERED=1
set UV_CONCURRENT_DOWNLOADS=1
set UV_PYTHON_INSTALL_DIR=D:\r2ai\py
set UV_CACHE_DIR=D:\r2ai\uvcache
set HF_HOME=D:\r2ai\hf
cd /d D:\r2ai
if not exist work\qrr06 mkdir work\qrr06
echo [%date% %time%] start > work\qrr06\setup.log
start "" /low /wait cmd /c "D:\r2ai\tools\uv.exe venv --python 3.11 D:\r2ai\venv_qwen >> work\qrr06\setup.log 2>&1"
echo [%date% %time%] venv done >> work\qrr06\setup.log
start "" /low /wait cmd /c "D:\r2ai\tools\uv.exe pip install --python D:\r2ai\venv_qwen\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0 >> work\qrr06\setup.log 2>&1"
echo [%date% %time%] torch done >> work\qrr06\setup.log
start "" /low /wait cmd /c "D:\r2ai\tools\uv.exe pip install --python D:\r2ai\venv_qwen\Scripts\python.exe numpy==1.26.4 transformers==4.51.3 tokenizers==0.21.1 huggingface_hub==0.30.2 safetensors==0.5.3 zstandard==0.23.0 >> work\qrr06\setup.log 2>&1"
echo [%date% %time%] libs done >> work\qrr06\setup.log
start "" /low /wait cmd /c "D:\r2ai\venv_qwen\Scripts\huggingface-cli.exe download Qwen/Qwen3-Reranker-0.6B --include *.json *.safetensors *.txt *.model >> work\qrr06\setup.log 2>&1"
echo [%date% %time%] model done >> work\qrr06\setup.log
D:\r2ai\venv_qwen\Scripts\python.exe -c "import torch,transformers;print(torch.__version__,transformers.__version__,torch.cuda.is_available())" >> work\qrr06\setup.log 2>&1
echo [%date% %time%] exit done >> work\qrr06\setup.log
