"""Read-only stage-C audit, writes one vi37 report; never touches crawl."""
import hashlib
import json
from pathlib import Path
import subprocess

OUT = Path('D:/r2ai/data/sub/vi37')


def main():
    result = {}
    for name in ['q8b_merge_report.json', 'q8b_select_round3.json', 'q8b_select_round4.json',
                 'chunk_report.json', 'build_report_vi37.json', 'stats_report_vi37.json']:
        p = OUT / name
        if p.exists():
            result[name] = json.loads(p.read_text(encoding='utf-8'))
    result['files'] = {}
    for r in (2, 3):
        for name in [f'q8b_r{r}_out.json', f'job_report_q8b_r{r}_out.json']:
            p = OUT / 'modal_out' / name
            if p.exists():
                row = {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                if name.startswith('job_report_'):
                    row['report'] = json.loads(p.read_text(encoding='utf-8'))
                result['files'][name] = row
    lock = Path('D:/r2ai/locks/heavy_job.lock')
    result['lock'] = lock.read_text() if lock.exists() else None
    task = subprocess.run(['powershell', '-NoProfile', '-Command',
        "Get-ScheduledTaskInfo -TaskName r2ai_vi37 | Select-Object LastRunTime,LastTaskResult | ConvertTo-Json"],
        capture_output=True, text=True)
    result['task_info'] = task.stdout.strip() or task.stderr.strip()
    (OUT / 'stage_c_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
