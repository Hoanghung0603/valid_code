@echo off
rem Recurring housekeeping task: deletes nodriver uc_* temp profile dirs whose
rem chrome process is gone (pool churn). Does NOT touch the crawler.
powershell -NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File D:\r2ai\data\crawl\tools\clean_temp_uc.ps1
exit /b
