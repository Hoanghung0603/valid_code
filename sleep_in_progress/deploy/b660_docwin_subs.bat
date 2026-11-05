@echo off
rem 4 bai nop ung vien tu chunk_dw_u* (doc = cache_pool_rr cat nguong logit -2.6, kep [40, 400]):
rem   sub_d600k60: chunk top-60 U=600   sub_d600k80: top-80 U=600
rem   sub_d900k60: chunk top-60 U=900   sub_d1200k50: top-50 U=1200
rem Moi bai: chunks_from_rerank -> union_sub -> evaluate.py validate -> zip (ZIP_DEFLATED, chi submission.json).
rem Log: D:\r2ai\data\sub\docwin\make_subs.log. Out: D:\r2ai\data\sub\docwin\submission_<ten>.json/.zip
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set DW=D:\r2ai\data\sub\docwin
echo ==== d600k60 ==== > %DW%\make_subs.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u600.json.zst --k 60 --out %DW%\chunks_d600k60.json >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d600k60.json --out %DW%\submission_d600k60.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d600k60.json >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d600k60.json --zip %DW%\submission_d600k60.zip >> %DW%\make_subs.log 2>&1
echo ==== d600k80 ==== >> %DW%\make_subs.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u600.json.zst --k 80 --out %DW%\chunks_d600k80.json >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d600k80.json --out %DW%\submission_d600k80.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d600k80.json >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d600k80.json --zip %DW%\submission_d600k80.zip >> %DW%\make_subs.log 2>&1
echo ==== d900k60 ==== >> %DW%\make_subs.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u900.json.zst --k 60 --out %DW%\chunks_d900k60.json >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d900k60.json --out %DW%\submission_d900k60.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d900k60.json >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d900k60.json --zip %DW%\submission_d900k60.zip >> %DW%\make_subs.log 2>&1
echo ==== d1200k50 ==== >> %DW%\make_subs.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u1200.json.zst --k 50 --out %DW%\chunks_d1200k50.json >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d1200k50.json --out %DW%\submission_d1200k50.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d1200k50.json >> %DW%\make_subs.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d1200k50.json --zip %DW%\submission_d1200k50.zip >> %DW%\make_subs.log 2>&1
echo exit done >> %DW%\make_subs.log
