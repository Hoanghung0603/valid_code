@echo off
if not exist "D:\r2ai\data\crawl\vi_topup_20261026" mkdir "D:\r2ai\data\crawl\vi_topup_20261026"
if not exist "D:\r2ai\data\crawl\vi_topup_20261026\audit" mkdir "D:\r2ai\data\crawl\vi_topup_20261026\audit"
cd /d D:\r2ai\data\crawl\tools
"D:\r2ai\.venv\Scripts\python.exe" crawl_vi_topup_20261026.py > D:\r2ai\data\crawl\async_recovery_20261009\audit\vi_topup_20261026.log 2> D:\r2ai\data\crawl\async_recovery_20261009\audit\vi_topup_20261026.err.log
