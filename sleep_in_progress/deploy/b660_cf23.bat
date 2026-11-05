@echo off
rem Bien the chunk_docfuse: s = zc + wd*zd, wd=2 (cf2) va wd=3 (cf3). Moi thu khac nhu b660_cf.bat (#5067).
rem Truoc het: wd=1 tai lap chunks_rrf3d2_cf.json (#5067), so nguyen van. KHONG ghi de file cua #5067.
rem Log: D:\r2ai\data\sub\diag10\cf23.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag10
set DW=D:\r2ai\data\sub\docwin
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set L=%W%\cf23.log
echo r2ai_cf23 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
echo ===== repro wd=1 >> %L%
%P% %S%\chunk_docfuse.py --rerank %DW%\chunk_dw_u900.json.zst --fused %W%\cache_rrf3d2.json --ref-chunks %DW%\chunks_d900t015.json --wd 1 --compare-with %W%\chunks_rrf3d2_cf.json --out %W%\chunks_rrf3d2_cf_repro.json >> %L% 2>&1
call :one cf2 2
call :one cf3 3
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:one
rem %1 = ten, %2 = wd
echo ===== %1 (wd %2) >> %L%
%P% %S%\chunk_docfuse.py --rerank %DW%\chunk_dw_u900.json.zst --fused %W%\cache_rrf3d2.json --ref-chunks %DW%\chunks_d900t015.json --wd %2 --compare-with %W%\chunks_rrf3d2_cf.json --out %W%\chunks_rrf3d2_%1.json >> %L% 2>&1
%P% %S%\full8b_sub.py --pool %POOL% --ens %W%\cache_rrf3d2.json --top-ens 1000000 --chunks-from %W%\chunks_rrf3d2_%1.json --ref %W%\submission_full_rrf3d2_cf.json --out %W%\submission_full_rrf3d2_%1.json >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_full_rrf3d2_%1.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_full_rrf3d2_%1.json --zip %W%\submission_full_rrf3d2_%1.zip --level 9 >> %L% 2>&1
%P% %S%\sub_stats.py %W%\submission_full_rrf3d2_%1.json >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %W%\cache_rrf3d2.json --ref %POOL% --n 30 --files %W%\submission_full_rrf3d2_%1.zip >> %L% 2>&1
goto :eof
