@echo off
rem Bai ung vien dot 4 (08/10): chunk theo nguong logit thich nghi, cung ngan sach zip (~90MB) voi 4806/4794.
rem   d900t015: U=900, logit > -0.15, kep [30, 110] (TB ~75, so voi d900k75 co dinh)
rem   d600t110: U=600, logit > -1.1, kep [40, 120] (TB ~99, so voi d600k100 co dinh)
rem Doc nhu d600k60 (cache_pool_rr, logit > -2.6, kep [40, 400]). Log: D:\r2ai\data\sub\docwin\make_subs4.log
set PYTHONIOENCODING=utf-8
set PYTHONUNBUFFERED=1
cd /d D:\r2ai\scripts
set P=D:\r2ai\.venv\Scripts\python.exe
set DW=D:\r2ai\data\sub\docwin
echo ==== d900t015 ==== > %DW%\make_subs4.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u900.json.zst --thr -0.15 --min-k 30 --max-k 110 --out %DW%\chunks_d900t015.json >> %DW%\make_subs4.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d900t015.json --out %DW%\submission_d900t015.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_subs4.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d900t015.json >> %DW%\make_subs4.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d900t015.json --zip %DW%\submission_d900t015.zip --level 9 >> %DW%\make_subs4.log 2>&1
echo ==== d600t110 ==== >> %DW%\make_subs4.log
%P% D:\r2ai\scripts\chunks_from_rerank.py --rerank %DW%\chunk_dw_u600.json.zst --thr -1.1 --min-k 40 --max-k 120 --out %DW%\chunks_d600t110.json >> %DW%\make_subs4.log 2>&1
%P% D:\r2ai\scripts\union_sub.py --docs-from D:\r2ai\data\sub\cache_pool_rr.json --chunks-from %DW%\chunks_d600t110.json --out %DW%\submission_d600t110.json --logit-thr -2.6 --min-n 40 --max-n 400 --docs-n 0 >> %DW%\make_subs4.log 2>&1
%P% D:\r2ai\scripts\evaluate.py validate %DW%\submission_d600t110.json >> %DW%\make_subs4.log 2>&1
%P% D:\r2ai\scripts\zip_submission.py --json %DW%\submission_d600t110.json --zip %DW%\submission_d600t110.zip --level 9 >> %DW%\make_subs4.log 2>&1
echo exit done >> %DW%\make_subs4.log
