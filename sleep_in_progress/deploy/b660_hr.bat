@echo off
rem Hop nhat rrf3d2 + HyDE + reason (fuse3 --extra), cong thuc #5248 (T -2.0, wd 4).
rem   hr: hyde x1 + reason x1 ; h2: hyde x2 + reason x1
rem Log: D:\r2ai\data\sub\diag12\hr.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W10=D:\r2ai\data\sub\diag10
set W11=D:\r2ai\data\sub\diag11
set W=D:\r2ai\data\sub\diag12
set H=D:\r2ai\data\sub\hyde
set DW=D:\r2ai\data\sub\docwin
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set Q8B=D:\r2ai\data\sub\qwen8b\cache_qwen8b_sorted.json
set DENSE=D:\r2ai\data\sub\zhprobe\dense_full_top200.json
set L=%W%\hr.log
echo r2ai_hr > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
call :one hr 1 1
call :one h2 2 1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:one
rem %1 = ten, %2 = w hyde, %3 = w reason
echo ===== %1 (hyde %2, reason %3) >> %L%
%P% %S%\fuse3.py --pool %POOL% --q8b %Q8B% --dense %DENSE% --w-bge 1 --w-q8b 1 --w-dense 2 --extra %H%\dense_hyde_top200.json %2 --extra %H%\dense_reason_top200.json %3 --extra-n 200 --k 60 --out %W%\cache_%1.json >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %W%\cache_%1.json --ref %W10%\cache_rrf3d2.json --n 30 >> %L% 2>&1
%P% %S%\chunk_docfuse.py --rerank %DW%\chunk_dw_u900.json.zst --fused %W%\cache_%1.json --ref-chunks %DW%\chunks_d900t015.json --wd 4 --compare-with %W11%\chunks_rrf3d2_cf4.json --out %W%\chunks_%1_cf4.json >> %L% 2>&1
%P% %S%\full8b_sub.py --pool %POOL% --ens %W%\cache_%1.json --top-ens 1000000 --thr -2.0 --chunks-from %W%\chunks_%1_cf4.json --ref %W11%\submission_t20_cf4.json --out %W%\submission_%1.json >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_%1.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_%1.json --zip %W%\submission_%1.zip --level 9 >> %L% 2>&1
goto :eof
