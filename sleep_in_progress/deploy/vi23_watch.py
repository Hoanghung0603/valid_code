"""Watch only the child GPU job; stop its process tree on repeated slow probes."""
import argparse
import subprocess
import time


def probe():
    try:
        result = subprocess.run(
            ["curl.exe", "-s", "--max-time", "15", "-o", "NUL", "-w",
             "%{http_code} %{time_total}", "https://www.vinmec.com/"],
            capture_output=True, text=True, timeout=20,
        )
        status, seconds = result.stdout.strip().split()
        elapsed = float(seconds)
        print(f"vinmec_watch {status} {elapsed:.6f}s", flush=True)
        return result.returncode == 0 and status != "000" and elapsed <= 1.0
    except (ValueError, subprocess.TimeoutExpired):
        print("vinmec_watch probe failed", flush=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing child command")
    if not probe():
        time.sleep(30)
        if not probe():
            return 2
    child = subprocess.Popen(command)
    print(f"watch child_pid={child.pid}", flush=True)
    next_probe = time.monotonic() + 30
    retry = False
    try:
        while child.poll() is None:
            if time.monotonic() >= next_probe:
                good = probe()
                if not good and retry:
                    print("vinmec repeated slow/failure: stopping own child tree", flush=True)
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False)
                    child.wait()
                    return 2
                retry = not good
                next_probe = time.monotonic() + 30
            time.sleep(1)
        return child.returncode
    finally:
        if child.poll() is None:
            subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], check=False)
            child.wait()


if __name__ == "__main__":
    raise SystemExit(main())
