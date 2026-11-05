@echo off
rem Bai ung vien dot 3 (08/10): so chunk dai vs nhieu chunk o cung kich thuoc zip (~<=95MB, upload 107MB bi treo).
rem   d900k75: U=900 top-75   d1200k60: U=1200 top-60   d600k100z9: d600k100 nen lai muc 9 (do muc giam)
rem Doc nhu d600k60 (cache_pool_rr, logit > -2.6, kep [40, 400]). Zip deflate muc 9.
rem Log: D:\r2ai\data\sub\docwin\make_subs3.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set DW=D:\r2ai\data\sub\docwin
echo ==== d900k75 ==== > %DW%\make_subs3.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u900.json.zst --k 75 --out %DW%\chunks_d900k75.json >> %DW%\make_subs3.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d900k75.json --out %DW%\submission_d900k75.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_subs3.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d900k75.json >> %DW%\make_subs3.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d900k75.json --zip %DW%\submission_d900k75.zip --level 9 >> %DW%\make_subs3.log 2>&1
echo ==== d1200k60 ==== >> %DW%\make_subs3.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u1200.json.zst --k 60 --out %DW%\chunks_d1200k60.json >> %DW%\make_subs3.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d1200k60.json --out %DW%\submission_d1200k60.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_subs3.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d1200k60.json >> %DW%\make_subs3.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d1200k60.json --zip %DW%\submission_d1200k60.zip --level 9 >> %DW%\make_subs3.log 2>&1
echo ==== d600k100z9 ==== >> %DW%\make_subs3.log
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d600k100.json --zip %DW%\submission_d600k100z9.zip --level 9 >> %DW%\make_subs3.log 2>&1
echo exit done >> %DW%\make_subs3.log
