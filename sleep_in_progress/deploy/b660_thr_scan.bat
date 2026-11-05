@echo off
rem Do nguong logit chunk thich nghi (08/10): moi cau lay chunk co logit > T, kep [min, max] -> phan bo ngan sach 100MB theo cau.
rem Chi tao file chunk de doc TB chunk/cau; chon T cho zip <= ~92MB roi moi dung bai.
rem Log: D:\r2ai\data\sub\docwin\thr_scan.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set DW=D:\r2ai\data\sub\docwin
echo ==== U600 ==== > %DW%\thr_scan.log
for %%T in (-0.6 -0.3 0.0) do (
  echo T=%%T >> %DW%\thr_scan.log
  %P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u600.json.zst --thr %%T --min-k 40 --max-k 120 --out %DW%\chunks_scan_u600.json >> %DW%\thr_scan.log 2>&1
)
echo ==== U900 ==== >> %DW%\thr_scan.log
for %%T in (-0.3 0.0 0.3) do (
  echo T=%%T >> %DW%\thr_scan.log
  %P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u900.json.zst --thr %%T --min-k 30 --max-k 110 --out %DW%\chunks_scan_u900.json >> %DW%\thr_scan.log 2>&1
)
echo exit done >> %DW%\thr_scan.log
