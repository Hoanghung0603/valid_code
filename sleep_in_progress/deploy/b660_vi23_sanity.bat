@echo off
rem vi23 sanity "khong-moi" (b660, CPU, ~2 phut): chay lai TOAN BO CHUOI bang du lieu cu (pool cu, dense cu,
rem HyDE cu, ung vien cu) -> phai tai lap submission_cx130cov.json 1200/1200 (doc + chunk) + rrf3d2/fhyde
rem trung y nguyen cache cu. Khong pass thi dung, khong dung bai chinh. Log: diag23\sanity.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag23
set L=%W%\sanity.log
if not exist %W% mkdir %W%

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_vi23_sanity" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_vi23_sanity > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === chain khong-moi >> %L%
%P% %S%\vi23.py chain --mode khong-moi --name km --pool D:\r2ai\data\sub\cache_pool_rr.json --q8b D:\r2ai\data\sub\qwen8b\cache_qwen8b_sorted.json --dense D:\r2ai\data\sub\zhprobe\dense_full_top200.json --hyde D:\r2ai\data\sub\hyde\dense_hyde_top200.json --cands D:\r2ai\data\sub\docwin_ext\chunk_dw_u900_ext.json.zst --ref-kq D:\r2ai\data\sub\diag14\submission_c120in.json --ref-chunks D:\r2ai\data\sub\diag11\chunks_rrf3d2_cf4.json --ref-sub D:\r2ai\data\sub\diag17\submission_cx130cov.json --out-dir %W% >> %L% 2>&1
if errorlevel 1 goto fail
echo === cmp-exact rrf3d2 >> %L%
%P% %S%\vi23.py cmp-exact --a %W%\cache_rrf3d2_khongmoi.json --b D:\r2ai\data\sub\diag10\cache_rrf3d2.json --report %W%\cmp_rrf3d2.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === cmp-exact fhyde >> %L%
%P% %S%\vi23.py cmp-exact --a %W%\cache_fhyde_khongmoi.json --b D:\r2ai\data\sub\diag12\cache_fhyde.json --report %W%\cmp_fhyde.json >> %L% 2>&1
if errorlevel 1 goto fail
echo === validate >> %L%
%P% %S%\evaluate.py validate %W%\submission_km.json >> %L% 2>&1
if errorlevel 1 goto fail
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo VI23_SANITY FAILED - khong duoc dung bai chinh >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
