@echo off
rem Do phu text kho URL moi (links_corpus_vi_en_zh) tren cac nguon text co san (b660, CPU).
rem Out: D:\r2ai\data\sub\diag21\{coverage.json,missing_vi_by_host.json,missing_zh_by_host.json,new_zh_ids.json}
rem Tao/xoa heavy_job.lock; neu lock dang co thi thoat (exit=locked). Log: %W%\coverage.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag21
set L=%W%\coverage.log
if not exist %W% mkdir %W%
if exist D:\r2ai\locks\heavy_job.lock (
  echo exit=locked >> %L%
  exit /b 1
)
echo coverage21 %date% %time% > D:\r2ai\locks\heavy_job.lock
echo === start %date% %time% > %L%
%P% %S%\coverage21.py --new D:\r2ai\data\sub\diag19\links_corpus_vi_en_zh.parquet --corpus-dir D:\r2ai\data\corpus\trunglt198 --vi-index-ids D:\r2ai\data\index\vi\ids.npy --zh-index-ids D:\r2ai\data\index\zh\ids.npy --zh-emb-parts D:\r2ai\data\sub\zhprobe --sub D:\r2ai\data\sub\diag17\submission_cx130cov.json --out-dir %W% >> %L% 2>&1
echo === exit %errorlevel% %date% %time% >> %L%
del D:\r2ai\locks\heavy_job.lock
echo exit done >> %L%
