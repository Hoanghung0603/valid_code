"""Nộp bài lên leaderboard Codabench bằng API (không cần trình duyệt) + chờ chấm + tải scoring_result.

    python scripts/submit_api.py submissions/v25.zip [--phase 44] [--wait]
Tài khoản đọc từ .r2ai_cred (R2AI_USER / R2AI_PASS). Kết quả chi tiết lưu data/eval/details_<id>_<tên>.jsonl.
"""
import argparse
import json
import io
import os
import re
import sys
import time
import zipfile

import requests

BASE = "https://leaderboard.aiguru.com.vn"


def session():
    cred = dict(l.strip().split("=", 1) for l in open(".r2ai_cred") if "=" in l)
    s = requests.Session()
    r = s.get(f"{BASE}/accounts/login/")
    tok = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text).group(1)
    s.post(f"{BASE}/accounts/login/", data={"csrfmiddlewaretoken": tok, "username": cred["R2AI_USER"],
                                           "password": cred["R2AI_PASS"]}, headers={"Referer": f"{BASE}/accounts/login/"})
    s.headers.update({"X-CSRFToken": s.cookies.get("csrftoken", ""), "Referer": f"{BASE}/competitions/15/"})
    return s


def submit(s, path, phase):
    name = os.path.basename(path)
    size = os.path.getsize(path)
    for attempt in range(6):   # PUT lớn đôi khi bị SSLEOF -> xin URL mới và thử lại
        nm = name if attempt == 0 else name.replace(".zip", f"_r{int(time.time())}.zip")   # tên trùng -> bị từ chối
        d = s.post(f"{BASE}/api/datasets/", json={"type": "submission", "request_sassy_file_name": nm, "name": nm,
                                                 "description": "", "is_public": False, "file_size": size}).json()
        if "key" not in d:
            print("dataset create rejected:", str(d)[:300], flush=True)
            continue
        key, url = d["key"], d["sassy_url"]
        t0 = time.time()
        if os.environ.get("R2AI_CURL"):   # curl (TLS stack khác, tự retry ở tầng TCP tốt hơn)
            import subprocess
            rc = subprocess.run(["curl", "-sS", "-f", "--http1.1", "--retry", "3", "--retry-all-errors", "-T", path, "-H", "Content-Type: application/zip", "-o", "/dev/null",
                                 "-w", "%{http_code} %{speed_upload}", url], capture_output=True, text=True)
            if rc.returncode == 0:
                print("curl upload ok", rc.stdout, f"{time.time() - t0:.0f}s", flush=True); break
            print(f"upload attempt {attempt + 1} failed after {time.time() - t0:.0f}s: curl {rc.returncode} {rc.stderr[:120]}", flush=True)
            time.sleep(30 * (attempt + 1)); continue
        try:
            with open(path, "rb") as f:
                r = requests.put(url, data=f, headers={"Content-Type": "application/zip", "Content-Length": str(size)},
                                 timeout=3600)
            r.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"upload attempt {attempt + 1} failed after {time.time() - t0:.0f}s: {type(e).__name__}", flush=True)
            time.sleep(30 * (attempt + 1))
    else:
        sys.exit("upload failed")
    s.put(f"{BASE}/api/datasets/completed/{key}/").raise_for_status()
    comp = s.get(f"{BASE}/api/competitions/15/").json()
    tasks = [t["id"] if isinstance(t, dict) else t for p in comp["phases"] if p["id"] == phase for t in p.get("tasks", [])]
    r = s.post(f"{BASE}/api/submissions/", json={"data": key, "phase": phase, "tasks": tasks, "fact_sheet_answers": None})
    r.raise_for_status()
    return r.json()["id"]


def prepare(s, path):
    """Chỉ tạo dataset + lấy presigned URL (để máy khác PUT file lên – xem kaggle_relay/)."""
    name = os.path.basename(path).replace(".zip", f"_{int(time.time())}.zip")
    d = s.post(f"{BASE}/api/datasets/", json={"type": "submission", "request_sassy_file_name": name, "name": name,
                                             "description": "", "is_public": False,
                                             "file_size": int(os.environ.get("R2AI_DECLARE_SIZE", os.path.getsize(path)))}).json()
    return d["key"], d["sassy_url"]


def finish(s, key, phase):
    s.put(f"{BASE}/api/datasets/completed/{key}/").raise_for_status()
    comp = s.get(f"{BASE}/api/competitions/15/").json()
    tasks = [t["id"] if isinstance(t, dict) else t for p in comp["phases"] if p["id"] == phase for t in p.get("tasks", [])]
    r = s.post(f"{BASE}/api/submissions/", json={"data": key, "phase": phase, "tasks": tasks, "fact_sheet_answers": None})
    r.raise_for_status()
    return r.json()["id"]


def wait(s, sid, name):
    while True:
        try:
            j = s.get(f"{BASE}/api/submissions/?phase__competition=15", timeout=120).json()
            st = next(x for x in (j.get("results") or j) if x["id"] == sid)["status"]
        except (requests.RequestException, ValueError, StopIteration) as e:   # mạng chập chờn -> thử lại
            print("poll error", type(e).__name__, flush=True); time.sleep(30); continue
        if st in ("Finished", "Failed", "Cancelled"):
            break
        time.sleep(20)
    if st != "Finished":
        print("status", st); return
    d = s.get(f"{BASE}/api/submissions/{sid}/get_details/").json()
    z = zipfile.ZipFile(io.BytesIO(requests.get(d["scoring_result"], timeout=300).content))
    print(z.read("scores.txt").decode())
    os.makedirs("data/eval", exist_ok=True)
    open(f"data/eval/details_{sid}_{name}.jsonl", "wb").write(z.read("details.jsonl"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("zip"); ap.add_argument("--phase", type=int, default=44); ap.add_argument("--wait", action="store_true")
    ap.add_argument("--fetch", type=int, default=0, help="chỉ chờ + tải kết quả của submission id đã nộp")
    ap.add_argument("--relay", action="store_true", help="tải lên qua Kaggle (kaggle_relay/) khi mạng tới BTC quá chậm")
    a = ap.parse_args()
    s = session()
    if a.fetch:
        wait(s, a.fetch, os.path.basename(a.zip).replace(".zip", "")); sys.exit()
    if a.relay:
        import shutil, subprocess
        key, url = prepare(s, a.zip)
        rd = "data/cache/relay_ds"; shutil.rmtree(rd, ignore_errors=True); os.makedirs(rd)
        tag = f"sub_{int(time.time())}.bin"; shutil.copy(a.zip, f"{rd}/{tag}"); open(f"{rd}/url.txt", "w").write(url)
        # 22/10: "datasets version" của r2ai-submit-relay lỗi Kaggle ("slugs and hashlink are all null") ⇒ mỗi lần tạo dataset MỚI, trỏ kernel vào nó
        slug = f"r2ai-relay-{int(time.time())}"; DS = f"kakaoffical/{slug}"
        open(f"{rd}/dataset-metadata.json", "w").write('{"title": "%s", "id": "%s", "licenses": [{"name": "other"}]}' % (slug, DS))
        kg = ".venv/bin/kaggle"; env = {k: v for k, v in os.environ.items() if k != "KAGGLE_API_TOKEN"}   # relay luôn ở kakaoffical
        r = subprocess.run([kg, "datasets", "create", "-p", rd], capture_output=True, text=True, env=env); print(r.stdout.strip()[-160:], r.stderr.strip()[-160:], flush=True)
        km = json.load(open("kaggle_relay/kernel-metadata.json")); km["dataset_sources"] = [DS]; json.dump(km, open("kaggle_relay/kernel-metadata.json", "w"))
        want = str(os.path.getsize(a.zip))
        for _ in range(180):   # chờ PHIÊN BẢN MỚI: sub.bin đúng dung lượng (status "ready" của bản cũ không đủ – v33 lần 1 bị Failed vì vậy)
            fl = subprocess.run([kg, "datasets", "files", DS], capture_output=True, text=True, env=env).stdout
            if any(l.split()[:2] == [tag, want] for l in fl.splitlines()):
                break
            time.sleep(20)
        else:
            sys.exit("relay dataset not updated")
        for i in range(10):   # 409 Conflict khi dataset mới chưa sẵn sàng -> thử lại
            r = subprocess.run([kg, "kernels", "push", "-p", "kaggle_relay"], capture_output=True, text=True, env=env)
            print(r.stdout.strip()[-200:], flush=True)
            if "successfully" in r.stdout:   # CLI trả 0 cả khi lỗi (vd. "Maximum batch CPU session count") ⇒ kiểm tra chữ
                break
            time.sleep(30)
        else:
            sys.exit("kernel push failed")
        time.sleep(30)
        while True:
            st = subprocess.run([kg, "kernels", "status", "kakaoffical/r2ai-relay-put"], capture_output=True, text=True, env=env).stdout
            if "COMPLETE" in st or "ERROR" in st:
                break
            time.sleep(15)
        shutil.rmtree("data/cache/relay_out", ignore_errors=True)   # tránh đọc put_result.txt cũ
        subprocess.run([kg, "kernels", "output", "kakaoffical/r2ai-relay-put", "-p", "data/cache/relay_out"], capture_output=True, env=env)
        res = open("data/cache/relay_out/put_result.txt").read() if os.path.exists("data/cache/relay_out/put_result.txt") else st
        print("relay:", res.strip(), flush=True)
        if tag not in res:   # kết quả phải của ĐÚNG lần này (tránh đọc output cũ – v47 lần 1 bị Failed vì vậy)
            sys.exit(f"relay result is stale/mismatched: {res[:120]}")
        if not res.startswith("200"):
            sys.exit("relay upload failed")
        sid = finish(s, key, a.phase)
    else:
        sid = submit(s, a.zip, a.phase)
    print("submitted id", sid, flush=True)
    if a.wait:
        wait(s, sid, os.path.basename(a.zip).replace(".zip", ""))
