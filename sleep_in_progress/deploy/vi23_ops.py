"""Read-only, compact vi23 status and deployment fingerprints over single-command SSH."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return {"bytes": path.stat().st_size, "sha256": digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["hashes", "status"])
    parser.add_argument("--base", default=r"D:\r2ai")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--machine", choices=["b660", "z87"])
    args = parser.parse_args()
    base = Path(args.base)
    if args.mode == "hashes":
        names = ["vi23.py", "docwin_run.py", "chunks_from_docwin.py", "dwx_merge.py",
                 "dwx_p3.py", "cxp3.py", "cx130.py", "evaluate.py", "zip_submission.py",
                 "split_file.py", "rerank_run.py", "chunking.py"]
        paths = {"scripts/" + name: base / "scripts" / name for name in names}
        for name in ["z87_vi23_docwin.bat", "z87_vi23_rr.bat", "b660_vi23_chunk.bat",
                     "b660_vi23_build.bat", "b660_vi23_stats.bat"]:
            paths[name] = base / "deploy" / name if args.local else base / name
        print(json.dumps({key: fingerprint(path) if path.exists() else None
                          for key, path in paths.items()}, indent=2))
        return
    work = base / ("work/vi23" if args.machine == "z87" else "data/sub/diag23")
    print("time", datetime.now().isoformat())
    for name in ["vi23_docwin.log", "chunk.log", "build.log", "stats.log"]:
        path = work / name
        if path.exists():
            print(name, "mtime", datetime.fromtimestamp(path.stat().st_mtime).isoformat(),
                  "bytes", path.stat().st_size)
            with path.open("rb") as handle:
                handle.seek(max(0, path.stat().st_size - 7000))
                lines = handle.read().decode("utf-8", errors="replace").splitlines()
            print("\n".join(line if len(line) < 1400 else line[:1000] + " [long line]"
                            for line in lines[-10:]))
    print("locks", {p.name: p.read_text(errors="replace").strip()
                    for p in (base / "locks").iterdir() if p.is_file()})
    tasks = subprocess.run(["schtasks", "/Query", "/FO", "CSV", "/NH"],
                           capture_output=True, text=True)
    print("vi23_tasks", "\n".join(line for line in tasks.stdout.splitlines() if "vi23" in line))
    ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                         "ConvertTo-Json -Compress -InputObject @(Get-CimInstance Win32_Process "
                         "-Filter \"Name='python.exe'\" -Property ProcessId,ParentProcessId,CommandLine,CreationDate)"],
                        capture_output=True, text=True)
    processes = json.loads(ps.stdout or "[]")
    if isinstance(processes, dict):
        processes = [processes]
    print("python_processes", json.dumps([
        {"ProcessId": row["ProcessId"], "ParentProcessId": row.get("ParentProcessId"),
         "CommandLine": (row.get("CommandLine") or "")[:260]}
        if "vi23" in (row.get("CommandLine") or "") else
        {"ProcessId": row["ProcessId"], "other_python": True} for row in processes]))
    if args.machine == "z87":
        subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,temperature.gpu",
                        "--format=csv,noheader"])


if __name__ == "__main__":
    main()
