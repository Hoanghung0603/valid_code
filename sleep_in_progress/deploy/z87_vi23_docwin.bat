@echo off
rem vi23 buoc 10 (z87, GPU bge-reranker-v2-m3): (1) sanity cham lai 300 cua so CU tu docwin_ext (lech logit
rem <= 1e-2); (2) docwin cho cap (q, doc) chua co ung vien chunk: cua so 300 tu buoc 250, max-len 512,
rem batch 16, GIONG docwin cu. Text: doc cu xuat tu index cu (bp cu) + doc moi (bp23). Resume theo phan
rem 50 cau. Do tre vinmec truoc moi buoc; > 1s sau 1 lan doi thi dung. Log: work\vi23\vi23_docwin.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
if exist D:\r2ai\locks\gpu.lock exit /b 2
echo r2ai_vi23_docwin > D:\r2ai\locks\gpu.lock
if not exist work\vi23\dw_parts mkdir work\vi23\dw_parts
set L=work\vi23\vi23_docwin.log
echo [%date% %time%] start vi23_docwin > %L%

call :vinmec
if errorlevel 1 goto fail
echo === sanity winsample 300 cua so cu >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dwx_p3.py winsample --docwin D:\r2ai\work\docwin_ext\docwin_ext.json.zst --docs D:\r2ai\work\docwin_ext\docwin_ext.part000.jsonl.zst D:\r2ai\work\docwin_ext\docwin_ext.part001.jsonl.zst --queries D:\r2ai\work\queries.jsonl --out D:\r2ai\work\vi23\winsample.json --n 300 >> %L% 2>&1"
if errorlevel 1 goto fail
echo === sanity score >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\vi23_watch.py -- D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dwx_p3.py score --in D:\r2ai\work\vi23\winsample.json --out D:\r2ai\work\vi23\winscore.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 >> %L% 2>&1"
if errorlevel 1 goto fail
echo === sanity wincheck >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dwx_p3.py wincheck --ref D:\r2ai\work\vi23\winsample.json --got D:\r2ai\work\vi23\winscore.json --report D:\r2ai\work\vi23\wincheck.json >> %L% 2>&1"
if errorlevel 1 goto fail

call :vinmec
if errorlevel 1 goto fail
set DOCS=
for %%f in (D:\r2ai\work\vi23\docs_vi23_dw_old*.jsonl.zst) do call set DOCS=%%DOCS%% %%f
for %%f in (D:\r2ai\work\vi23\docs_vi23_dw_new*.jsonl.zst) do call set DOCS=%%DOCS%% %%f
echo === docwin_run cap moi >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\vi23_watch.py -- D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\docwin_run.py --docs %DOCS% --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\vi23\cands_docwin_vi23.json --out D:\r2ai\work\vi23\docwin_vi23.json.zst --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top-docs 500 --window-words 300 --window-stride 250 --resume-dir D:\r2ai\work\vi23\dw_parts >> %L% 2>&1"
if errorlevel 1 goto fail

call :vinmec
if errorlevel 1 goto fail
del D:\r2ai\locks\gpu.lock
echo [%date% %time%] DONE >> %L%
echo exit done >> %L%
goto :eof

:vinmec
curl.exe -s -o NUL -w "vinmec: %%{http_code} %%{time_total}s" https://www.vinmec.com/ >> %L% 2>&1
echo. >> %L%
curl.exe -s -o NUL -w "%%{time_total}" https://www.vinmec.com/ > work\vi23\vm.txt 2>&1
findstr /b /c:"0." work\vi23\vm.txt >nul
if not errorlevel 1 goto vmok
echo vinmec > 1s, doi 30s >> %L%
ping -n 31 localhost >nul 2>&1
curl.exe -s -o NUL -w "%%{time_total}" https://www.vinmec.com/ > work\vi23\vm.txt 2>&1
findstr /b /c:"0." work\vi23\vm.txt >nul
if not errorlevel 1 goto vmok
echo vinmec van > 1s, dung job >> %L%
exit /b 1
:vmok
type work\vi23\vm.txt >> %L%
echo. >> %L%
exit /b 0

:fail
del D:\r2ai\locks\gpu.lock
echo [%date% %time%] VI23_DOCWIN FAILED >> %L%
echo exit fail >> %L%
exit /b 1
