import subprocess, os
open("/kaggle/working/r.bin","wb").write(os.urandom(5_000_000))
for ip in ["", "104.21.7.227", "172.67.156.130"]:
    args = ["curl", "-s", "-o", "/dev/null", "-m", "60", "-w", "%{speed_upload} B/s %{http_code} %{time_total}s", "-X", "PUT", "--data-binary", "@/kaggle/working/r.bin"]
    if ip: args += ["--resolve", f"leaderboard.aiguru.com.vn:443:{ip}"]
    print(ip or "dns", subprocess.run(args + ["https://leaderboard.aiguru.com.vn/private/dataset/speedtest.bin"], capture_output=True, text=True).stdout, flush=True)
