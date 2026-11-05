@echo off
rem vi23 buoc 7 (z87, GPU bge-reranker-v2-m3): (1) sanity cham lai 200 cap da co logit (logit lech <= 1e-2);
rem (2) rerank cap (q, doc MOI) thieu logit trong pool_v23 (fp16, max-len 512, batch 16, text = title +
rem sach 3.000 ky tu). Do tre vinmec truoc moi buoc; > 1s sau 1 lan doi thi dung. Log: work\vi23\vi23_rr.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
set L=work\vi23\vi23_rr.log
echo [%date% %time%] start vi23_rr > %L%

call :vinmec
if errorlevel 1 goto fail
echo === sanity rerank 200 cap cu >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\rerank_run.py --docs D:\r2ai\work\all\docs_vi_all.part000.jsonl.zst D:\r2ai\work\all\docs_vi_all.part001.jsonl.zst D:\r2ai\work\all\docs_vi_all.part002.jsonl.zst D:\r2ai\work\all\docs_vi_all.part003.jsonl.zst D:\r2ai\work\all\docs_vi_all.part004.jsonl.zst D:\r2ai\work\all\docs_vi_all.part005.jsonl.zst D:\r2ai\work\all\docs_vi_all.part006.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\vi23\sanity_pairs.json --out D:\r2ai\work\vi23\sanity_out.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top 50 --resume-dir D:\r2ai\work\vi23\rr_sanity_parts >> %L% 2>&1"
if errorlevel 1 goto fail
echo === cmplogit sanity >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\vi23.py cmplogit --got D:\r2ai\work\vi23\sanity_out.json --ref D:\r2ai\work\vi23\sanity_ref.json --tol 0.01 --report D:\r2ai\work\vi23\sanity_cmplogit.json >> %L% 2>&1"
if errorlevel 1 goto fail

call :vinmec
if errorlevel 1 goto fail
echo === rerank cap doc moi >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\rerank_run.py --docs D:\r2ai\work\vi23\docs_embed\docs_vi23_embed.jsonl.zst --queries D:\r2ai\work\queries.jsonl --cands D:\r2ai\work\vi23\cands_rr_new.json --out D:\r2ai\work\vi23\rerank_new.json --model BAAI/bge-reranker-v2-m3 --max-len 512 --batch 16 --top 500 --resume-dir D:\r2ai\work\vi23\rr_new_parts >> %L% 2>&1"
if errorlevel 1 goto fail

call :vinmec
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
echo [%date% %time%] VI23_RR FAILED >> %L%
echo exit fail >> %L%
goto :eof
