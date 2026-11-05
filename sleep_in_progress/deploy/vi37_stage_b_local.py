"""Local file checks / explicit per-file Volume transfer for vi37 stage B. No GPU launch."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

STAGE = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi37')
CLI = __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'


def modal(*args):
    subprocess.run([CLI, *args, '--profile', __import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>")], check=True)


def fingerprint(path):
    return {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['upload', 'bench-get', 'shards-get', 'verify', 'zip'])
    a = ap.parse_args()
    if a.action == 'upload':
        manifest = json.loads((STAGE / 'stage_a_verification.json').read_text(encoding='utf-8'))
        for item in manifest['files']:
            if '\\q8b_input\\' not in item['path']:
                continue
            name = item['path'].rsplit('\\', 1)[-1]
            local = STAGE / 'q8b_input' / name
            assert fingerprint(local) == {k: item[k] for k in ('bytes', 'sha256')}, name
            if name == 'queries.jsonl':
                assert fingerprint(local) == fingerprint(STAGE / 'volume_queries_before_b.jsonl')
            print('VERIFIED', name, fingerprint(local), flush=True)
        for name in manifest['q8b_export_report.json']['files']:
            modal('volume', 'put', '--force', 'r2ai-runner', str(STAGE / 'q8b_input' / name), '/in/vi37/' + name)
        modal('volume', 'ls', 'r2ai-runner', '/in/vi37/')
    elif a.action in ('bench-get', 'shards-get'):
        names = ['q8b_bench_out'] if a.action == 'bench-get' else ['q8b_shard0_out', 'q8b_shard1_out']
        for stem in names:
            for name in [stem + '.json', 'job_report_' + stem + '.json']:
                modal('volume', 'get', 'r2ai-runner', '/out/vi37/' + name, str(STAGE / 'modal_out' / name))
    elif a.action == 'verify':
        report = json.loads((STAGE / 'stage_b_verification.json').read_text(encoding='utf-8'))
        for row in report['modal_out']:
            assert fingerprint(STAGE / 'modal_out' / row['name']) == {
                k: row[k] for k in ('bytes', 'sha256')}, row['name']
        gpu_s = sum(row['report']['elapsed'] for row in report['modal_out'] if 'report' in row)
        result = {'output_hashes_match_b660': True, 'stage_b_gpu_seconds': gpu_s,
                  'stage_b_gpu_usd_est': gpu_s * 1.95 / 3600,
                  'stage_b_plus30pct_usd_est': gpu_s * 1.95 / 3600 * 1.3,
                  'vi37_successful_reports_gpu_seconds': 2992 + gpu_s,
                  'vi37_successful_reports_gpu_usd_est': (2992 + gpu_s) * 1.95 / 3600,
                  'note': 'Billing not verified; stage-A app-wall conservative upper bound separately in handoff.'}
        (STAGE / 'stage_b_local_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
    else:
        path = STAGE / 'submission_vi37.zip'
        report = json.loads((STAGE / 'build_report_vi37.json').read_text(encoding='utf-8'))
        fp = fingerprint(path)
        assert fp['bytes'] == report['zip_bytes'] <= 99_000_000
        assert fp['sha256'] == report['sha256']
        with zipfile.ZipFile(path) as z:
            assert z.namelist() == ['submission.json']
            assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
            sub = json.loads(z.read('submission.json'))
        assert len(sub) == 1200 and {r['id'] for r in sub} == set(range(1, 1201))
        assert all(c['doc_id'] in r['relevant_docs'] for r in sub for c in r['relevant_chunks'])
        fp.update(queries=1200, entries=1, lzma=True, chunk_doclist_violations=0)
        (STAGE / 'local_verification_b.json').write_text(json.dumps(fp, indent=2), encoding='utf-8')
        print(json.dumps(fp), flush=True)


if __name__ == '__main__':
    main()
