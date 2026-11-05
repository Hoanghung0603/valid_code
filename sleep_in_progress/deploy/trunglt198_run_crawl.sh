#!/bin/bash
# Crawler tren trunglt-198 (IP cong ty). Chay tach khoi phien SSH: setsid -f ${HOME}/r2ai/run_crawl.sh
# Dung em: touch ${HOME}/r2ai/data/crawl/STOP ; nho xoa STOP truoc khi chay lai. Resume tu dong.
cd ${HOME}/r2ai || exit 1
exec nohup .venv/bin/python scripts/crawl.py --out data/crawl --tiers 1,2,3 --exclude-hosts cnkang.com,120ask.com \
  --concurrency 96 --per-host 1 --big-per-host 4 --max-mbps 200 --nice 10 --extract-workers 8 --min-free-gb 5 \
  --report-every 120 --log-file data/crawl/crawl.log > data/crawl/stdout.log 2>&1 < /dev/null
