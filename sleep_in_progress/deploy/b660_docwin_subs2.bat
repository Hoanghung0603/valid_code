@echo off
rem Bai ung vien dot 2 (08/10): chunk top-100 / top-120 U=600 (doc nhu d600k60: cache_pool_rr, logit > -2.6, kep [40, 400]).
rem Log: D:\r2ai\data\sub\docwin\make_subs2.log. Out: D:\r2ai\data\sub\docwin\submission_<ten>.json/.zip
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set DW=D:\r2ai\data\sub\docwin
echo ==== d600k100 ==== > %DW%\make_subs2.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u600.json.zst --k 100 --out %DW%\chunks_d600k100.json >> %DW%\make_subs2.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d600k100.json --out %DW%\submission_d600k100.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_subs2.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d600k100.json >> %DW%\make_subs2.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d600k100.json --zip %DW%\submission_d600k100.zip >> %DW%\make_subs2.log 2>&1
echo ==== d600k120 ==== >> %DW%\make_subs2.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u600.json.zst --k 120 --out %DW%\chunks_d600k120.json >> %DW%\make_subs2.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d600k120.json --out %DW%\submission_d600k120.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_subs2.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d600k120.json >> %DW%\make_subs2.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d600k120.json --zip %DW%\submission_d600k120.zip >> %DW%\make_subs2.log 2>&1
echo exit done >> %DW%\make_subs2.log
