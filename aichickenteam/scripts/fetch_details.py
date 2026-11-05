"""Tải details.jsonl của bài đã chấm khi scoring_result có thể xuất hiện muộn: thử lại mỗi --every giây, tối đa --tries lần.
    python scripts/fetch_details.py 6575 v74_zh"""
import io, sys, time, zipfile, argparse
import requests
sys.path.insert(0, "scripts")
from submit_api import session, BASE
ap = argparse.ArgumentParser(); ap.add_argument("sid", type=int); ap.add_argument("name"); ap.add_argument("--every", type=int, default=300); ap.add_argument("--tries", type=int, default=36); a = ap.parse_args()
for t in range(a.tries):
    try:
        d = session().get(f"{BASE}/api/submissions/{a.sid}/get_details/", timeout=60).json()
        if d.get("scoring_result"):
            z = zipfile.ZipFile(io.BytesIO(requests.get(d["scoring_result"], timeout=300).content))
            open(f"data/eval/details_{a.sid}_{a.name}.jsonl", "wb").write(z.read("details.jsonl")); print("OK", a.sid, a.name, flush=True); sys.exit(0)
        print(time.strftime("%H:%M"), "chưa có scoring_result", flush=True)
    except Exception as e:
        print("lỗi", type(e).__name__, flush=True)
    time.sleep(a.every)
sys.exit(1)
