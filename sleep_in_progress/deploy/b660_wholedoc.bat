@echo off
rem Thi nghiem chunk = toan van doc (08/10). Thu tu doc: rr (rerank doc) va dw (max cua so docwin); ngan sach ky tu/cau 200k, 250k.
rem Doc nhu d600k60 (cache_pool_rr, logit > -2.6, kep [40, 400]). Chi validate + zip; chon bai theo kich thuoc zip (< ~95MB).
rem Log: D:\r2ai\data\sub\wd\make_wd.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set WD=D:\r2ai\data\sub\wd
set C100=D:\r2ai\data\sub\chunkdocs100
if not exist %WD% mkdir %WD%
%P% D:\r2ai\scripts\chunks_wholedoc.py --docs %C100%\chunkdocs100.part000.jsonl.zst %C100%\chunkdocs100.part001.jsonl.zst --cands %C100%\cands_chunkdocs100.json --docwin D:\r2ai\data\sub\docwin\docwin_top100.json.zst --orders rr dw --budgets 200000 250000 --doc-cap 40000 --out-prefix %WD%\chunks_wd > %WD%\make_wd.log 2>&1
for %%N in (rr_b200k rr_b250k dw_b200k dw_b250k) do (
  echo ==== %%N ==== >> %WD%\make_wd.log
  %P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %WD%\chunks_wd_%%N.json --out %WD%\submission_wd_%%N.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %WD%\make_wd.log 2>&1
  %P% D:\r2ai\scripts\evaluate.py validate %WD%\submission_wd_%%N.json >> %WD%\make_wd.log 2>&1
  %P% D:\r2ai\scripts\zip_submission.py --json %WD%\submission_wd_%%N.json --zip %WD%\submission_wd_%%N.zip --level 9 >> %WD%\make_wd.log 2>&1
)
echo exit done >> %WD%\make_wd.log
