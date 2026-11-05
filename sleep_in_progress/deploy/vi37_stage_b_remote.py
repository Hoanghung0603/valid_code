"""Small read-only audit of vi37 stage-B reports and transferred files on b660."""
import hashlib
import json
from pathlib import Path
import sys

OUT = Path('D:/r2ai/data/sub/vi37')


def main():
    result = {}
    for name in ['q8b_merge_report.json', 'q8b_select_round2.json', 'chunk_report.json',
                 'build_report_vi37.json', 'stats_report_vi37.json']:
        path = OUT / name
        if path.exists():
            result[name] = json.loads(path.read_text(encoding='utf-8'))
    result['modal_out'] = []
    for stem in ['q8b_bench_out', 'q8b_shard0_out', 'q8b_shard1_out']:
        for name in [stem + '.json', 'job_report_' + stem + '.json']:
            path = OUT / 'modal_out' / name
            if path.exists():
                row = {'name': name, 'bytes': path.stat().st_size,
                       'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
                if name.startswith('job_report_'):
                    row['report'] = json.loads(path.read_text(encoding='utf-8'))
                result['modal_out'].append(row)
    result['lock'] = Path('D:/r2ai/locks/heavy_job.lock').exists()
    text = json.dumps(result, ensure_ascii=False, indent=2)
    (OUT / 'stage_b_verification.json').write_text(text, encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
