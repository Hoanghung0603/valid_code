@echo off
rem Xuat text 540.863 doc cua chi muc ZH (tieu de + text sach, cat 1.500 ky tu ~ 512 token) de embed bge-m3 tren z87.
rem Out: D:\r2ai\data\sub\zhprobe\docs_zh_all.partNNN.jsonl.zst (<=80MB). Log: D:\r2ai\data\sub\zhprobe\export_zh.log
rem Tao/xoa lock heavy_job.lock; neu lock dang co thi thoat ngay (exit=locked).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set W=D:\r2ai\data\sub\zhprobe
set L=D:\r2ai\locks\heavy_job.lock
if not exist %W% mkdir %W%
if exist %L% (
  echo exit=locked >> %W%\export_zh.log
  exit /b 1
)
echo zhprobe b660_zh_export %date% %time% > %L%
cd /d D:\r2ai\scripts
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\export_docs.py --all --index D:\r2ai\data\index\zh --chars 1500 --out-docs %W%\docs_zh_all.jsonl.zst --part-mb 80 > %W%\export_zh.log 2>&1
echo exit=%ERRORLEVEL% >> %W%\export_zh.log
del %L%
