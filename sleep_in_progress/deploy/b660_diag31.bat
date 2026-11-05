@echo off
rem diag31: docceil tren pool vi30 — relevant_docs = toan bo pool da loc (fhyde truoc,
rem pool sau) + doc vi30/chunk thieu noi cuoi, chunk y nguyen vi30. Zip LZMA, cat chunk
rem cuoi moi cau neu zip > 99.000.000 B. Khong nop bai (nop thu cong qua leaderboard).
rem Log: D:\r2ai\data\sub\diag31\diag31.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag31
set D30=D:\r2ai\data\sub\diag30
set L=%W%\diag31.log
if not exist %W% mkdir %W%
echo r2ai_diag31 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
echo === docceil >> %L%
%P% %S%\diag31.py docceil --sub %D30%\submission_vi30.json --pool %D30%\in1\pool_v30.json --fhyde %D30%\cache_fhyde_v30.json --out %W%\submission_docceil30.json --zip %W%\submission_docceil30.zip --report %W%\docceil_report.json >> %L% 2>&1
echo === validate >> %L%
%P% %S%\evaluate.py validate %W%\submission_docceil30.json >> %L% 2>&1
echo === sha256 >> %L%
certutil -hashfile %W%\submission_docceil30.zip SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
