@echo off
rem ZH dense tren tap giao (id ZH kho moi x doc ZH da embed) + RRF k60 giua truy VI va ZH.
rem z87 GPU uu tien thap, lock D:\r2ai\locks\gpu.lock (neu da co thi exit=locked).
rem Out: D:\r2ai\work\zhnew\zhnew_{vi,zh,rrf}_top100.json + zhnew_meta.json. Log: %W%\zhnew.log
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
set W=D:\r2ai\work\zhnew
if not exist D:\r2ai\locks mkdir D:\r2ai\locks
if not exist %W% mkdir %W%
if exist D:\r2ai\locks\gpu.lock (
  echo exit=locked >> %W%\zhnew.log
  exit /b 1
)
echo zhnew %date% %time% > D:\r2ai\locks\gpu.lock
echo [%date% %time%] start >> %W%\zhnew.log
curl.exe -s -o NUL -w "vinmec_before=%%{http_code} %%{time_total}s" https://www.vinmec.com/ >> %W%\zhnew.log
echo. >> %W%\zhnew.log
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\zhnew_dense.py --emb D:\r2ai\work\zh\emb --zh-ids %W%\new_zh_ids.json --queries D:\r2ai\work\queries.jsonl --zh-queries %W%\zh_queries.jsonl --out-vi %W%\zhnew_vi_top100.json --out-zh %W%\zhnew_zh_top100.json --out-rrf %W%\zhnew_rrf_top100.json --out-meta %W%\zhnew_meta.json --top 100 --k 60 >> %W%\zhnew.log 2>&1"
curl.exe -s -o NUL -w "vinmec_after=%%{http_code} %%{time_total}s" https://www.vinmec.com/ >> %W%\zhnew.log
echo. >> %W%\zhnew.log
echo [%date% %time%] DONE >> %W%\zhnew.log
del D:\r2ai\locks\gpu.lock
echo exit done >> %W%\zhnew.log
