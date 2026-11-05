@echo off
rem Thi nghiem "rai chunk theo doc" (08/10): chunk tot nhat cua moi doc truoc -> phu nhieu doc khac nhau hon.
rem   d600s100: U=600 spread top-100 (doi chung 4794 d600k100)   d900s75: U=900 spread top-75 (doi chung 4806 d900k75)
rem Doc nhu d600k60. Log: D:\r2ai\data\sub\docwin\make_spread.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set DW=D:\r2ai\data\sub\docwin
echo ==== d600s100 ==== > %DW%\make_spread.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u600.json.zst --k 100 --spread --out %DW%\chunks_d600s100.json >> %DW%\make_spread.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d600s100.json --out %DW%\submission_d600s100.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_spread.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d600s100.json >> %DW%\make_spread.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d600s100.json --zip %DW%\submission_d600s100.zip --level 9 >> %DW%\make_spread.log 2>&1
echo ==== d900s75 ==== >> %DW%\make_spread.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u900.json.zst --k 75 --spread --out %DW%\chunks_d900s75.json >> %DW%\make_spread.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d900s75.json --out %DW%\submission_d900s75.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_spread.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d900s75.json >> %DW%\make_spread.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d900s75.json --zip %DW%\submission_d900s75.zip --level 9 >> %DW%\make_spread.log 2>&1
%P% D:\r2ai\scripts\sub_stats.py %DW%\submission_d600s100.json %DW%\submission_d900s75.json >> %DW%\make_spread.log 2>&1
echo exit done >> %DW%\make_spread.log
