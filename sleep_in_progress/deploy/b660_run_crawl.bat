@echo off
rem Crawler tren b660 (IP nha, chung Wi-Fi voi z87). Chay qua Task Scheduler: schtasks /Run /TN r2ai_crawl
rem Dung em: tao file D:\r2ai\data\crawl\STOP ; nho xoa file STOP truoc khi chay lai.
rem 04/10: tang big-per-host 4 -> 8, concurrency 8 -> 16 (gap doi toc do), giu tran 20 Mbit/s va uu tien thap.
cd /d D:\r2ai
D:\r2ai\.venv\Scripts\python.exe D:\r2ai\scripts\crawl.py --out data/crawl --hosts cnkang.com,120ask.com --per-host 1 --big-per-host 8 --concurrency 16 --max-mbps 20 --nice 1 --extract-workers 3 --min-free-gb 50 --report-every 60 --log-file data/crawl/crawl.log
