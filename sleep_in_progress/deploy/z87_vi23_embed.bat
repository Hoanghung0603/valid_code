@echo off
rem vi23 buoc 4 (z87, GPU bge-m3, uu tien thap): (1) sanity embed lai 200 doc cu -> cmp-emb voi emb da luu
rem (cosine >= 0.999); (2) embed 54.164 doc MOI (title + sach 3.000 ky tu, CLS fp16 512 token) -> work\emb23;
rem (3) top-200 cosine chi tren doc MOI cho cau goc (256) va HyDE (512). Do tre vinmec truoc moi buoc;
rem > 1s sau 1 lan doi thi dung job. Log: work\vi23\vi23_embed.log
set PYTHONUNBUFFERED=1
set HF_HOME=D:\r2ai\hf
set HF_HUB_OFFLINE=1
cd /d D:\r2ai
if not exist work\vi23 mkdir work\vi23
if not exist work\emb23 mkdir work\emb23
if not exist work\emb23san mkdir work\emb23san
if not exist work\emb_empty mkdir work\emb_empty
set L=work\vi23\vi23_embed.log
echo [%date% %time%] start vi23_embed > %L%
if not exist work\vi23\docs_embed mkdir work\vi23\docs_embed
if not exist work\vi23\docs_san mkdir work\vi23\docs_san
if exist work\vi23\docs_vi23_embed.jsonl.zst move /y work\vi23\docs_vi23_embed.jsonl.zst work\vi23\docs_embed\docs_vi23_embed.jsonl.zst >> %L% 2>&1

call :vinmec
if errorlevel 1 goto fail
echo === lay 200 doc cu lam mau >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\vi23.py sample-docs --docs D:\r2ai\work\all --n 200 --out D:\r2ai\work\vi23\docs_san\san200.jsonl.zst --report D:\r2ai\work\vi23\sample200.json >> %L% 2>&1"
if errorlevel 1 goto fail
echo === sanity embed 200 doc cu >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\embed_all.py --docs D:\r2ai\work\vi23\docs_san --work-old D:\r2ai\work\emb_empty --work D:\r2ai\work\emb23san --queries D:\r2ai\work\queries.jsonl --out D:\r2ai\work\vi23\dense_sanity_ignore.json --doc-len 512 --query-len 256 --batch 16 --top 5 >> %L% 2>&1"
if errorlevel 1 goto fail
echo === cmp-emb sanity >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\vi23.py cmp-emb --new-dir D:\r2ai\work\emb23san --old-dirs D:\r2ai\work\emb D:\r2ai\work\emb_all --tol 0.999 --report D:\r2ai\work\vi23\cmp_emb_sanity.json >> %L% 2>&1"
if errorlevel 1 goto fail

call :vinmec
if errorlevel 1 goto fail
echo === embed 54k doc moi >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\embed_all.py --docs D:\r2ai\work\vi23\docs_embed --work-old D:\r2ai\work\emb_empty --work D:\r2ai\work\emb23 --queries D:\r2ai\work\queries.jsonl --out D:\r2ai\work\vi23\dense_scan_ignore.json --doc-len 512 --query-len 256 --batch 16 --top 5 >> %L% 2>&1"
if errorlevel 1 goto fail

call :vinmec
if errorlevel 1 goto fail
echo === dense top-200 tren doc moi (cau goc 256) >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dense_topk.py --emb D:\r2ai\work\emb23 --docs D:\r2ai\work\vi23\docs_embed\docs_vi23_embed.jsonl.zst --queries D:\r2ai\work\queries.jsonl --query-len 256 --top 200 --chunk 131072 --device cuda --out D:\r2ai\work\vi23\dense_new_top200.json >> %L% 2>&1"
if errorlevel 1 goto fail
echo === dense top-200 tren doc moi (HyDE 512) >> %L%
start "" /low /wait cmd /c "D:\r2ai\venv\Scripts\python.exe D:\r2ai\scripts\dense_topk.py --emb D:\r2ai\work\emb23 --docs D:\r2ai\work\vi23\docs_embed\docs_vi23_embed.jsonl.zst --queries D:\r2ai\work\hyde\q_hyde.jsonl --query-len 512 --top 200 --chunk 131072 --device cuda --out D:\r2ai\work\vi23\hyde_new_top200.json >> %L% 2>&1"
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
echo [%date% %time%] VI23_EMBED FAILED >> %L%
echo exit fail >> %L%
goto :eof
