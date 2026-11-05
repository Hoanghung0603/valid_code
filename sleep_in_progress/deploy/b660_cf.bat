@echo off
rem Chon lai chunk: s = z(logit chunk U900) + z(diem doc rrf3d2), k_q = so chunk cua d900t015.
rem relevant_docs = n_q doc dau theo rrf3d2 (adaptive_n pool -2.6 [40,400]) + doc nguon chunk moi.
rem Log: D:\r2ai\data\sub\diag10\cf.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag10
set DW=D:\r2ai\data\sub\docwin
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set L=%W%\cf.log
echo r2ai_cf > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%
echo === chunk_docfuse >> %L%
%P% %S%\chunk_docfuse.py --rerank %DW%\chunk_dw_u900.json.zst --fused %W%\cache_rrf3d2.json --ref-chunks %DW%\chunks_d900t015.json --out %W%\chunks_rrf3d2_cf.json >> %L% 2>&1
echo === full8b_sub >> %L%
%P% %S%\full8b_sub.py --pool %POOL% --ens %W%\cache_rrf3d2.json --top-ens 1000000 --chunks-from %W%\chunks_rrf3d2_cf.json --ref %W%\submission_full_rrf3d2.json --out %W%\submission_full_rrf3d2_cf.json >> %L% 2>&1
echo === validate >> %L%
%P% %S%\evaluate.py validate %W%\submission_full_rrf3d2_cf.json >> %L% 2>&1
echo === zip >> %L%
%P% %S%\zip_submission.py --json %W%\submission_full_rrf3d2_cf.json --zip %W%\submission_full_rrf3d2_cf.zip --level 9 >> %L% 2>&1
%P% %S%\sub_stats.py %W%\submission_full_rrf3d2_cf.json >> %L% 2>&1
%P% %S%\bm25_chunk.py report --cache %W%\cache_rrf3d2.json --ref %POOL% --n 30 --files %W%\submission_full_rrf3d2_cf.zip >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
