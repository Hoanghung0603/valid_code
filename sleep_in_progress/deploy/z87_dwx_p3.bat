@echo off
rem Phan B1 (z87, GPU bge-reranker-v2-m3): (1) sanity cham lai 300 cua so CU tu docwin_ext
rem (winsample/score/wincheck, logit khop ~1e-2); (2) docwin mo rong cho (q, doc) cua 3 doclist
rem t20/t16/nq chua co ung vien chunk - cua so 300 tu buoc 250, max-len 512, batch 16, GIONG
rem docwin cu. Resume theo phan 50 cau. Do do tre vinmec truoc/sau; dung neu > 1s lap lai.
rem Out: D:\r2ai\work\docwin_p3\docwin_p3.json.zst. Log: work\docwin_p3\dwx_p3.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
if not exist work\docwin_p3 mkdir work\docwin_p3
set L=work\docwin_p3\dwx_p3.log
echo [%date% %time%] start dwx_p3 > %L%
echo === vinmec truoc >> %L%
curl.exe -s -o NUL -w "vinmec: %%{http_code} %%{time_total}s" https://www.vinmec.com/ >> %L% 2>&1
echo. >> %L%

echo === sanity winsample >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dwx_p3.py winsample --docwin D:\r2ai\work\docwin_ext\docwin_ext.json.zst --docs D:\r2ai\work\docwin_ext\docwin_ext.part000.jsonl.zst D:\r2ai\work\docwin_ext\docwin_ext.part001.jsonl.zst --queries D:\r2ai\work\queries.jsonl --out D:\r2ai\work\docwin_p3\winsample.json --n 300 >> %L% 2>&1"
if errorlevel 1 goto fail
echo === sanity score >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dwx_p3.py score --in D:\r2ai\work\docwin_p3\winsample.json --out D:\r2ai\work\docwin_p3\winscore.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 >> %L% 2>&1"
if errorlevel 1 goto fail
echo === sanity wincheck >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dwx_p3.py wincheck --ref D:\r2ai\work\docwin_p3\winsample.json --got D:\r2ai\work\docwin_p3\winscore.json --report D:\r2ai\work\docwin_p3\wincheck.json >> %L% 2>&1"
if errorlevel 1 goto fail

echo === vinmec truoc docwin >> %L%
curl.exe -s -o NUL -w "vinmec: %%{http_code} %%{time_total}s" https://www.vinmec.com/ >> %L% 2>&1
echo. >> %L%
set DOCS=
for %%f in (D:\r2ai\work\docwin_p3\docwin_p3.part*.jsonl.zst) do call set DOCS=%%DOCS%% %%f
echo === docwin_run mo rong >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\docwin_run.py --docs %DOCS% --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\docwin_p3\cands_docwin_p3.json --out D:\r2ai\work\docwin_p3\docwin_p3.json.zst --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top-docs 500 --window-words 300 --window-stride 250 --resume-dir D:\r2ai\work\docwin_p3\dwx_parts >> %L% 2>&1"
if errorlevel 1 goto fail
echo === vinmec sau >> %L%
curl.exe -s -o NUL -w "vinmec: %%{http_code} %%{time_total}s" https://www.vinmec.com/ >> %L% 2>&1
echo. >> %L%
echo [%date% %time%] DONE >> %L%
echo exit done >> %L%
goto :eof

:fail
echo [%date% %time%] DWX_P3 FAILED >> %L%
echo exit fail >> %L%
goto :eof
