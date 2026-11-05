#!/bin/bash
# Nộp lại có kiểm tra trùng: trước mỗi lần thử, hỏi API xem đã có bài cùng tiền tố tên chưa.
#   bash scripts/submit_retry.sh submissions/v69_probe_zh.zip [số lần=8] [phase=44]
cd /home/phong/Desktop/hsk/test2
Z=$1; N=${2:-8}; PH=${3:-44}; PRE=$(basename $Z .zip)
for i in $(seq 1 $N); do
  if timeout 120 .venv/bin/python - "$PRE" <<'PY'
import sys; sys.path.insert(0, "scripts")
from submit_api import session, BASE
s = session(); j = s.get(f"{BASE}/api/submissions/?phase__competition=15", timeout=60).json()
L = j if isinstance(j, list) else j.get("results", j)
hit = [x for x in L if (x.get("filename") or x.get("name") or "").startswith(sys.argv[1])]
print("đã có:", [(x["id"], x["status"]) for x in hit]); sys.exit(0 if hit else 1)
PY
  then echo "ĐÃ NỘP – dừng (lấy kết quả bằng --wait thủ công nếu cần)"; exit 0; fi
  echo "== lần $i $(date '+%H:%M')"
  R2AI_CURL=1 timeout 5400 .venv/bin/python scripts/submit_api.py $Z --phase $PH --wait 2>&1 | grep -vE "^\s|^Traceback" | tail -12
  [ ${PIPESTATUS[0]} -eq 0 ] && exit 0
  sleep 300
done
exit 1
