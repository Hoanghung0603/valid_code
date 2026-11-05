#!/bin/bash
# Đẩy lần lượt kaggle_crawl3/s0..s7 (giữ tối đa 4 phiên CPU crawl ⇒ còn 1 slot cho relay nộp bài), rồi tải output về data/cache/crawl3/s<k>.
export KAGGLE_API_TOKEN=$(cat ~/.kaggle/access_token)
KG=.venv/bin/kaggle
running() { local n=0 x; for x in crawl2-s5 crawl2-s6 crawl2-s7 $(for k in 0 1 2 3 4 5 6 7; do echo crawl3-s$k; done); do
  $KG kernels status kakaoffical/r2ai-$x 2>/dev/null | grep -qE "RUNNING|QUEUED" && n=$((n+1)); done; echo $n; }
collect() { local k=$1; [ -d data/cache/crawl3/s$k/docs_out ] && return; $KG kernels status kakaoffical/r2ai-crawl3-s$k 2>/dev/null | grep -qE "COMPLETE|ERROR" || return
  $KG kernels output kakaoffical/r2ai-crawl3-s$k -p data/cache/crawl3/s$k >/dev/null 2>&1; echo "$(date +%H:%M) s$k collected: $(ls data/cache/crawl3/s$k/docs_out 2>/dev/null | wc -l) parquet"; }
pushed="${PUSHED:-}"   # đã đẩy trước đó (vd. PUSHED=0 START=1)
for k in ${START:-0} $(seq $(( ${START:-0} + 1 )) 7); do
  while [ "$(running)" -ge 4 ]; do for j in $pushed; do collect $j; done; sleep 600; done
  until r=$($KG kernels push -p kaggle_crawl3/s$k 2>&1 | tail -1); echo "$r" | grep -q successfully; do echo "push s$k failed: $r"; sleep 600; done
  echo "$(date +%H:%M) pushed s$k"; pushed="$pushed $k"; sleep 120
done
for j in $pushed; do until [ -d data/cache/crawl3/s$j/docs_out ]; do collect $j; [ -d data/cache/crawl3/s$j/docs_out ] || sleep 600; done; done
echo "crawl3 all collected"
