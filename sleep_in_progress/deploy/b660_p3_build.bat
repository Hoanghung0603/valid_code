@echo off
rem p3 build (CPU b660): sanity chi tieu -> cache_pool3_rr -> mo rong nguon dense/HyDE -> cache_fhyde3
rem -> doclist T -2.0 va -1.6. SANITY BAT BUOC:
rem   (1) cham lai 200 cap sanity (cmplogit <= 1e-2)
rem   (2) cung code tren pool cu tai lap cache_fhyde (top-137 >= 99% cau) + doc submission_split
rem Log: D:\r2ai\data\sub\diag18\p3_build.log. Lock: D:\r2ai\locks\heavy_job.lock (tao dau, xoa cuoi).
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag18
set W11=D:\r2ai\data\sub\diag11
set W12=D:\r2ai\data\sub\diag12
set POOL=D:\r2ai\data\sub\cache_pool_rr.json
set P2=%W%\cache_pool2_rr.json
set HYDE=D:\r2ai\data\sub\hyde\dense_hyde_top200.json
set REASON=D:\r2ai\data\sub\hyde\dense_reason_top200.json
set DENSE400=%W%\dense_full_top400.json
set BM25=D:\r2ai\data\sub\cache_vi_k300.json
set DENSE200=D:\r2ai\data\sub\zhprobe\dense_full_top200.json
set Q8B=D:\r2ai\data\sub\qwen8b\cache_qwen8b_sorted.json
set L=%W%\p3_build.log
if not exist %W% mkdir %W%

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_p3build" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_p3build > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === sanity rerank 200 cap >> %L%
%P% %S%\pool3.py cmplogit --got %W%\sanity_out.json --ref %W%\sanity_ref.json --tol 0.01 --report %W%\sanity_cmplogit.json >> %L% 2>&1
if errorlevel 1 goto fail

echo === sanity tai lap cache_fhyde tren pool cu >> %L%
%P% %S%\fuse3.py --pool %POOL% --q8b %Q8B% --dense %DENSE200% --w-bge 1 --w-q8b 1 --w-dense 2 --bm25 %HYDE% --w-bm25 1 --bm25-n 200 --k 60 --out %W%\cache_fhyde_re.json >> %L% 2>&1
if errorlevel 1 goto fail
%P% %S%\pool3.py cmp --a %W12%\cache_fhyde.json --b %W%\cache_fhyde_re.json --top 137 --min-ratio 0.99 --report %W%\cmp_fhyde_re.json >> %L% 2>&1
if errorlevel 1 goto fail

echo === sanity doc submission_split >> %L%
%P% %S%\full8b_sub.py --pool %POOL% --ens %W%\cache_fhyde_re.json --top-ens 1000000 --thr -2.0 --chunks-from %W11%\chunks_rrf3d2_cf4.json --ref %W11%\submission_t20_cf4.json --out %W%\submission_split_re.json >> %L% 2>&1
if errorlevel 1 goto fail
%P% %S%\pool3.py cmpsub --a %W%\submission_split_re.json --b %W12%\submission_split.json --min-ratio 0.99 --report %W%\cmp_split_re.json >> %L% 2>&1
if errorlevel 1 goto fail

echo === cache_pool3_rr >> %L%
%P% %S%\pool3.py pool3 --pool %POOL% --pool2 %P2% --new %W%\new_cands.json --rerank %W%\rerank_p3.json --idx-from %HYDE% %REASON% %DENSE400% %BM25% --out %W%\cache_pool3_rr.json --report %W%\pool3_report.json >> %L% 2>&1
if errorlevel 1 goto fail

echo === extend nguon dense va HyDE >> %L%
%P% %S%\pool3.py extend --base %DENSE200% --new-scores %W%\cache_new_dense.json --out %W%\cache_dense3.json --report %W%\extend_dense.json >> %L% 2>&1
if errorlevel 1 goto fail
%P% %S%\pool3.py extend --base %HYDE% --new-scores %W%\cache_new_hyde.json --out %W%\cache_hyde3.json --report %W%\extend_hyde.json >> %L% 2>&1
if errorlevel 1 goto fail

echo === cache_fhyde3 >> %L%
%P% %S%\fuse3.py --pool %W%\cache_pool3_rr.json --q8b %Q8B% --dense %W%\cache_dense3.json --w-bge 1 --w-q8b 1 --w-dense 2 --bm25 %W%\cache_hyde3.json --w-bm25 1 --bm25-n 1000 --k 60 --out %W%\cache_fhyde3.json >> %L% 2>&1
if errorlevel 1 goto fail

echo === doclist T -2.0 va -1.6 >> %L%
%P% %S%\pool3.py doclist --pool3 %W%\cache_pool3_rr.json --rank %W%\cache_fhyde3.json --new %W%\new_cands.json --meta %W%\new_cands_meta.json --thr -2.0 --lo 40 --hi 400 --out %W%\doclist_p3_t20.json --report %W%\doclist_p3_t20_report.json >> %L% 2>&1
if errorlevel 1 goto fail
%P% %S%\pool3.py doclist --pool3 %W%\cache_pool3_rr.json --rank %W%\cache_fhyde3.json --new %W%\new_cands.json --meta %W%\new_cands_meta.json --thr -1.6 --lo 40 --hi 400 --out %W%\doclist_p3_t16.json --report %W%\doclist_p3_t16_report.json >> %L% 2>&1
if errorlevel 1 goto fail

echo === thong ke doi chieu fhyde3 voi fhyde (thong tin, khong chan) >> %L%
%P% %S%\pool3.py cmp --a %W12%\cache_fhyde.json --b %W%\cache_fhyde3.json --top 137 --min-ratio 0 --report %W%\cmp_fhyde3_vs_fhyde.json >> %L% 2>&1
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo P3BUILD FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
