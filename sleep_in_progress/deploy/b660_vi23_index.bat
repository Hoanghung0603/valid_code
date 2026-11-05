@echo off
rem vi23 buoc 1 (b660, CPU): dung chi muc BM25 GOP (cu + moi) ra D:\r2ai\data\index\vi23 (KHONG dong vao
rem index\vi). Shard = 59 shard cu (D:\r2ai\data\corpus\trunglt198, giu thu tu) + 3 shard moi da copy.
rem boilerplate.json gop da co san (reuse_bp). B1 tuong minh ~6 phut, gop ~5-10 phut. Merge tung that bai
rem OSError Errno 22 (lan dung index cu 05/10) -> thu lai 1 lan (parts co done marker nen chi chay lai gop).
rem Sau do verify idx cu giu nguyen (ids.npy[:613087] == ids.npy index cu) + new_idx.json.
rem Log: D:\r2ai\data\sub\diag23\index.log. Lock heavy_job.lock.
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag23
set IX=D:\r2ai\data\index\vi23
set L=%W%\index.log
if not exist %IX% mkdir %IX%

:waitlock
if not exist D:\r2ai\locks\heavy_job.lock goto havelock
findstr /c:"r2ai_vi23_index" D:\r2ai\locks\heavy_job.lock >nul 2>&1
if not errorlevel 1 goto havelock
echo doi lock cua job khac %TIME% >> %L%
ping -n 31 localhost >nul 2>&1
goto waitlock
:havelock
echo r2ai_vi23_index > D:\r2ai\locks\heavy_job.lock
echo start %DATE% %TIME% > %L%

echo === bigindex build vi23 >> %L%
%P% %S%\bigindex.py build --shards D:\r2ai\data\corpus\trunglt198 %W%\shards --out %IX% --workers 4 >> %L% 2>&1
if not errorlevel 1 goto built
echo === loi build, thu lai sau 60s (parts co done marker, chi chay lai gop) >> %L%
ping -n 61 localhost >nul 2>&1
%P% %S%\bigindex.py build --shards D:\r2ai\data\corpus\trunglt198 %W%\shards --out %IX% --workers 4 >> %L% 2>&1
if not errorlevel 1 goto built
goto fail
:built
echo === verify idx >> %L%
%P% %S%\vi23.py verify-idx --index-old-ids D:\r2ai\data\index\vi\ids.npy --index-new-ids %IX%\ids.npy --new-ids %W%\new_ids.json --out %W%\new_idx.json --report %W%\verify_idx_report.json >> %L% 2>&1
if errorlevel 1 goto fail
del D:\r2ai\locks\heavy_job.lock
echo end %DATE% %TIME% >> %L%
echo exit done >> %L%
goto :eof

:fail
echo VI23_INDEX FAILED >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit fail >> %L%
goto :eof
