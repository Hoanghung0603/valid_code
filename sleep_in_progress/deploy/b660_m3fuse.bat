@echo off
rem Them nguon bge-m3 ColBERT / sparse (cham lai pool tren z87, cache_m3_*.json) vao hop nhat rrf3d2.
rem Dung khe nguon thu 4 cua fuse3.py (--bm25 ... --bm25-n 1000 = lay ca pool).
rem Moi bien the: fuse3 -> chunk_docfuse wd 4 -> full8b_sub thr -2.0 (cong thuc #5248) -> validate -> zip.
rem Log: D:\r2ai\data\sub\diag12\m3fuse.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W10=D:\r2ai\data\sub\diag10
set W11=D:\r2ai\data\sub\diag11
set W=D:\r2ai\data\sub\diag12
set M3=D:\r2ai\data\sub\m3
set DW=D:\r2ai\data\sub\docwin
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set Q8B=D:\r2ai\data\sub\qwen8b\cache_qwen8b_sorted.json
set DENSE=D:\r2ai\data\sub\zhprobe\dense_full_top200.json
set L=%W%\m3fuse.log
if not exist %W% mkdir %W%
echo r2ai_m3fuse > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
echo ===== novelty: top30 m3 vs dense, vs bge, vs rrf3d2 >> %L%
%P% %S%\bm25_chunk.py report --cache %M3%\cache_m3_colbert.json --ref %DENSE% --n 30 >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %M3%\cache_m3_colbert.json --ref %POOL% --n 30 >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %M3%\cache_m3_colbert.json --ref %W10%\cache_rrf3d2.json --n 30 >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %M3%\cache_m3_sparse.json --ref %DENSE% --n 30 >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %M3%\cache_m3_sparse.json --ref %W10%\cache_rrf3d2.json --n 30 >> %L% 2>&1
call :one c1 cache_m3_colbert 1
call :one c2 cache_m3_colbert 2
call :one s1 cache_m3_sparse 1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:one
rem %1 = ten, %2 = cache m3, %3 = trong so
echo ===== %1 (%2 w %3) >> %L%
%P% %S%\fuse3.py --pool %POOL% --q8b %Q8B% --dense %DENSE% --w-bge 1 --w-q8b 1 --w-dense 2 --bm25 %M3%\%2.json --w-bm25 %3 --bm25-n 1000 --k 60 --out %W%\cache_%1.json >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %W%\cache_%1.json --ref %W10%\cache_rrf3d2.json --n 30 >> %L% 2>&1
%P% %S%\chunk_docfuse.py --rerank %DW%\chunk_dw_u900.json.zst --fused %W%\cache_%1.json --ref-chunks %DW%\chunks_d900t015.json --wd 4 --compare-with %W11%\chunks_rrf3d2_cf4.json --out %W%\chunks_%1_cf4.json >> %L% 2>&1
%P% %S%\full8b_sub.py --pool %POOL% --ens %W%\cache_%1.json --top-ens 1000000 --thr -2.0 --chunks-from %W%\chunks_%1_cf4.json --ref %W11%\submission_t20_cf4.json --out %W%\submission_%1.json >> %L% 2>&1
%P% %S%\evaluate.py validate %W%\submission_%1.json >> %L% 2>&1
%P% %S%\zip_submission.py --json %W%\submission_%1.json --zip %W%\submission_%1.zip --level 9 >> %L% 2>&1
goto :eof
