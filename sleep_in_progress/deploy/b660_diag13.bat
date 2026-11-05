@echo off
rem diag13: (1) do doc ngoai top-100 bge, (2) zip DEFLATED9/BZIP2/LZMA + kiem doc lai,
rem (3) docceil/chunkceil, (4) nqf theo diem fused fhyde. Khong nop bai.
rem Log: D:\r2ai\data\sub\diag13\diag13.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag13
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set FH=D:\r2ai\data\sub\diag12\cache_fhyde.json
set SPLIT=D:\r2ai\data\sub\diag12\submission_split.json
set DW=D:\r2ai\data\sub\docwin\chunk_dw_u900.json.zst
set L=%W%\diag13.log
if not exist %W% mkdir %W%
echo r2ai_diag13 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
echo === job1 >> %L%
%P% %S%\diag13.py job1 --split %SPLIT% --pool %POOL% --fhyde %FH% --out %W%\job1_report.json >> %L% 2>&1
echo === job2 >> %L%
%P% %S%\diag13.py job2 --split %SPLIT% --out-dir %W% --report %W%\job2_report.json >> %L% 2>&1
echo === docceil >> %L%
%P% %S%\diag13.py docceil --split %SPLIT% --pool %POOL% --fhyde %FH% --out %W%\submission_docceil.json --report %W%\docceil_report.json >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_docceil.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_docceil.json --zip %W%\submission_docceil.zip --level 9 >> %L% 2>&1
%P% %S%\sub_stats.py %W%\submission_docceil.json >> %L% 2>&1
echo === chunkceil >> %L%
%P% %S%\diag13.py chunkceil --split %SPLIT% --rerank %DW% --out %W%\submission_chunkceil.json --report %W%\chunkceil_report.json --zip-limit 95000000 --probe-zip %W%\submission_chunkceil.probe.zip >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_chunkceil.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_chunkceil.json --zip %W%\submission_chunkceil.zip --level 9 >> %L% 2>&1
%P% %S%\sub_stats.py %W%\submission_chunkceil.json >> %L% 2>&1
echo === nqf >> %L%
%P% %S%\diag13.py nqf --split %SPLIT% --pool %POOL% --fhyde %FH% --out %W%\submission_nqf.json --report %W%\nqf_report.json --target 137 >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_nqf.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_nqf.json --zip %W%\submission_nqf.zip --level 9 >> %L% 2>&1
%P% %S%\sub_stats.py %W%\submission_nqf.json >> %L% 2>&1
echo === sha256 >> %L%
certutil -hashfile %W%\submission_docceil.zip SHA256 >> %L% 2>&1
certutil -hashfile %W%\submission_chunkceil.zip SHA256 >> %L% 2>&1
certutil -hashfile %W%\submission_nqf.zip SHA256 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
