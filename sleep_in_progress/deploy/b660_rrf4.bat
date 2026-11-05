@echo off
rem Them BM25 vao hop nhat doc (rrf3d2 + BM25 top-300 trong so w) va thu RRF k=30.
rem Moi bien the: fuse3 -> chunk_docfuse wd=3 -> full8b_sub (n_q nhu #5084) -> validate -> zip.
rem Truoc het: w_bm25=0 phai tai lap cache_rrf3d2.json (so bang fc /b).
rem Log: D:\r2ai\data\sub\diag11\rrf4.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W10=D:\r2ai\data\sub\diag10
set W=D:\r2ai\data\sub\diag11
set DW=D:\r2ai\data\sub\docwin
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set Q8B=D:\r2ai\data\sub\qwen8b\cache_qwen8b_sorted.json
set DENSE=D:\r2ai\data\sub\zhprobe\dense_full_top200.json
set BM=D:\r2ai\data\sub\cache_vi_k300.json
set L=%W%\rrf4.log
if not exist %W% mkdir %W%
echo r2ai_rrf4 > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
echo ===== repro w_bm25=0 >> %L%
%P% %S%\fuse3.py --pool %POOL% --q8b %Q8B% --dense %DENSE% --w-bge 1 --w-q8b 1 --w-dense 2 --bm25 %BM% --w-bm25 0 --k 60 --out %W%\cache_repro.json >> %L% 2>&1
fc /b %W%\cache_repro.json %W10%\cache_rrf3d2.json > nul
if errorlevel 1 (echo REPRO KHAC >> %L%) else (echo REPRO GIONG >> %L%)
call :one b05 0.5 60
call :one b1 1 60
call :one k30 0 30
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:one
rem %1 = ten, %2 = w_bm25, %3 = k
echo ===== %1 (w_bm25 %2, k %3) >> %L%
%P% %S%\fuse3.py --pool %POOL% --q8b %Q8B% --dense %DENSE% --w-bge 1 --w-q8b 1 --w-dense 2 --bm25 %BM% --w-bm25 %2 --k %3 --out %W%\cache_%1.json >> %L% 2>&1
echo === top30 overlap %1 vs rrf3d2 >> %L%
%P% %S%\bm25_chunk.py report --cache %W%\cache_%1.json --ref %W10%\cache_rrf3d2.json --n 30 >> %L% 2>&1
%P% %S%\chunk_docfuse.py --rerank %DW%\chunk_dw_u900.json.zst --fused %W%\cache_%1.json --ref-chunks %DW%\chunks_d900t015.json --wd 3 --compare-with %W10%\chunks_rrf3d2_cf3.json --out %W%\chunks_%1_cf3.json >> %L% 2>&1
%P% %S%\full8b_sub.py --pool %POOL% --ens %W%\cache_%1.json --top-ens 1000000 --chunks-from %W%\chunks_%1_cf3.json --ref %W10%\submission_full_rrf3d2_cf3.json --out %W%\submission_%1_cf3.json >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_%1_cf3.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_%1_cf3.json --zip %W%\submission_%1_cf3.zip --level 9 >> %L% 2>&1
%P% %S%\sub_stats.py %W%\submission_%1_cf3.json >> %L% 2>&1
goto :eof
