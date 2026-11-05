"""CPU-only vi38 artefact/status audit; may run on b660 or a local export directory."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('D:/r2ai'))
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding='utf-8')
    root = a.root/'data/sub/vi38'
    report = {}
    for name in ('select_report.json', 'export_report.json', 'build_report_sanity.json', 'stats_report_sanity.json'):
        p = root/name
        if p.exists():
            ob = json.loads(p.read_text(encoding='utf-8'))
            report[name] = {k: v for k, v in ob.items() if k not in ('files', 'modes', 'build')}
    lock = a.root/'locks/heavy_job.lock'
    report['lock'] = lock.read_text() if lock.exists() else None
    result = subprocess.run(['schtasks', '/Query', '/TN', 'r2ai_vi38', '/FO', 'LIST', '/V'],
                            capture_output=True, text=True)
    report['task'] = result.stdout
    report['task_exit'] = result.returncode
    report['artifacts'] = {p.name: {'bytes': p.stat().st_size, 'mtime': p.stat().st_mtime}
                           for p in root.glob('submission_sanity.*')}
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
