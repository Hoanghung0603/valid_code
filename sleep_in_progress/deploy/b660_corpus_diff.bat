@echo off
rem So sanh kho URL cu/moi cua BTC va do ti le doc bai nop con nam trong kho moi.
set P=D:\r2ai\.venv\Scripts\python.exe
set S=D:\r2ai\scripts
set W=D:\r2ai\data\sub\diag19
set L=%W%\corpus_diff.log
if not exist %W% mkdir %W%
if exist D:\r2ai\locks\heavy_job.lock (
  echo lock dang co, dung >> %L%
  exit /b 1
)
echo corpus_diff > D:\r2ai\locks\heavy_job.lock
echo === start %date% %time% >> %L%
set PYTHONIOENCODING=utf-8
%P% %S%\corpus_diff.py --old D:\r2ai\data\links_corpus.parquet --new %W%\links_corpus_vi_en_zh.parquet --sub D:\r2ai\data\sub\diag17\submission_cx130cov.json --sub D:\r2ai\data\sub\diag18\submission_ceil2.json --out %W%\corpus_diff.json >> %L% 2>&1
echo === exit %errorlevel% %date% %time% >> %L%
del D:\r2ai\locks\heavy_job.lock
