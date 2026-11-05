#!/bin/bash
# Nộp 1 bài vào VÒNG PRIVATE (phase 45, task 51; 01/11 07:00 → 04/11 07:00 GMT+7; CHỈ 5 BÀI TỔNG).
#   bash scripts/private_submit.sh submissions/private/P1_v62.zip
# Thử nộp thẳng (curl) 2 lần, rồi relay Kaggle. Ghi log data/cache/tmp/logs/private_<tên>.log
set -o pipefail
cd /home/phong/Desktop/hsk/test2
Z=$1; [ -s "$Z" ] || { echo "thiếu file $Z"; exit 1; }
L=data/cache/tmp/logs/private_$(basename $Z .zip).log
grep -q "$(sha256sum $Z | cut -d' ' -f1)" submissions/private/SHA256SUMS || { echo "SHA256 không khớp SHA256SUMS – file đã đổi?"; exit 1; }
for i in 1 2; do
  echo "== thẳng lần $i $(date '+%m-%d %H:%M')" | tee -a $L
  R2AI_CURL=1 .venv/bin/python scripts/submit_api.py $Z --phase 45 --wait 2>&1 | tee -a $L | tail -10 && exit 0
  grep -q "submitted id" $L && { echo "ĐÃ có submitted id – KHÔNG nộp lại (tránh mất lượt)"; exit 1; }
  sleep 60
done
echo "== relay $(date '+%m-%d %H:%M')" | tee -a $L
.venv/bin/python scripts/submit_api.py $Z --phase 45 --relay --wait 2>&1 | tee -a $L | tail -10
