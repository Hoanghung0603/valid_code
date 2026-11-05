#!/bin/bash
# Chờ IP local hết bị Long Châu chặn (thử 1 URL mỗi giờ), rồi crawl chậm ids_r3_8 (--delay 8 ≈ 450 trang/giờ).
U=$(sqlite3 data/cache/crawl3_local/crawl.sqlite "select url from crawl where status='blocked' limit 1")
until [ "$(.venv/bin/python -c "from curl_cffi import requests; print(requests.get('$U',impersonate='chrome',timeout=30).status_code)" 2>/dev/null)" = 200 ]; do sleep 3600; done
echo "$(date +%H:%M) local unblocked, crawling ids_r3_8"
nice .venv/bin/python -m src.crawl --ids-file data/cache/ids_r3_8.csv --per-domain 1 --delay 8 --concurrency 4 --flush-every 1000 \
  --out-dir data/cache/crawl3_local8/docs_out --state-db data/cache/crawl3_local8/crawl.sqlite > data/cache/crawl3_local8.log 2>&1
echo "$(date +%H:%M) local crawl done: $(tail -1 data/cache/crawl3_local8.log | cut -c1-200)"
