@echo off
rem 2 luot cuoi ngay 11/10, moi bai doi dung 1 bien so voi #5124 (T -2.0, chunk wd 3):
rem   t18: T -1.8, chunk nhu #5124 (chunks_rrf3d2_cf3.json)
rem   t20w4: T -2.0, chunk chon lai voi wd 4
rem Log: D:\r2ai\data\sub\diag11\last2.log. Lock: D:\r2ai\locks\heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W10=D:\r2ai\data\sub\diag10
set W=D:\r2ai\data\sub\diag11
set DW=D:\r2ai\data\sub\docwin
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set L=%W%\last2.log
echo r2ai_last2 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
echo ===== t18 >> %L%
%P% %S%\full8b_sub.py --pool %POOL% --ens %W10%\cache_rrf3d2.json --top-ens 1000000 --thr -1.8 --chunks-from %W10%\chunks_rrf3d2_cf3.json --ref %W%\submission_t20_cf3.json --out %W%\submission_t18_cf3.json >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_t18_cf3.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_t18_cf3.json --zip %W%\submission_t18_cf3.zip --level 9 >> %L% 2>&1
echo ===== t20w4 >> %L%
%P% %S%\chunk_docfuse.py --rerank %DW%\chunk_dw_u900.json.zst --fused %W10%\cache_rrf3d2.json --ref-chunks %DW%\chunks_d900t015.json --wd 4 --compare-with %W10%\chunks_rrf3d2_cf3.json --out %W%\chunks_rrf3d2_cf4.json >> %L% 2>&1
%P% %S%\full8b_sub.py --pool %POOL% --ens %W10%\cache_rrf3d2.json --top-ens 1000000 --thr -2.0 --chunks-from %W%\chunks_rrf3d2_cf4.json --ref %W%\submission_t20_cf3.json --out %W%\submission_t20_cf4.json >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_t20_cf4.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_t20_cf4.json --zip %W%\submission_t20_cf4.zip --level 9 >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
