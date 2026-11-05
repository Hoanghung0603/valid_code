"""Bounded supplementary 8B transfers/checks; no submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

STAGE = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi37')
CLI = __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'


def modal(*args):
    env = dict(os.environ, PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
    return subprocess.run([CLI, *args, '--profile', __import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>")], check=True, env=env)


def fp(p):
    return {'bytes': p.stat().st_size, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=['upload', 'get', 'verify', 'zip', 'apps', 'audit'])
    ap.add_argument('--round', type=int, choices=[2, 3], default=2)
    a = ap.parse_args()
    r = a.round
    if a.action == 'audit':
        remote = json.loads((STAGE / 'stage_c_verification.json').read_text(encoding='utf-8'))
        local = json.loads((STAGE / 'local_round2_verification.json').read_text(encoding='utf-8'))
        for name, expected in local['files'].items():
            assert fp(STAGE / 'modal_out' / name) == expected
            assert expected == {k: remote['files'][name][k] for k in ('bytes', 'sha256')}
        assert remote['lock'] is None
        assert remote['q8b_select_round3.json']['pairs'] == 0
        assert remote['build_report_vi37.json']['verbatim_bad'] == 0
        sec = local['gpu_seconds']
        result = {'output_hashes_match_b660': True, 'stage_c_gpu_seconds': sec,
                  'stage_c_gpu_usd_est': sec * 1.95 / 3600,
                  'stage_c_app_wall_seconds': 45,
                  'stage_c_wall_plus30pct_usd_est': 45 * 1.95 / 3600 * 1.3,
                  'total_successful_gpu_seconds': 3205.9 + sec,
                  'total_successful_gpu_usd_est': (3205.9 + sec) * 1.95 / 3600,
                  'total_app_wall_seconds_including_failed_A': 5498 + 45,
                  'total_wall_plus30pct_usd_est': (5498 + 45) * 1.95 / 3600 * 1.3,
                  'billing_verified': False}
        (STAGE / 'stage_c_local_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
        return
    if a.action == 'zip':
        report = json.loads((STAGE / 'build_report_vi37.json').read_text(encoding='utf-8'))
        path = STAGE / 'submission_vi37.zip'
        result = fp(path)
        assert result['bytes'] == report['zip_bytes'] <= 99_000_000
        assert result['sha256'] == report['sha256']
        with zipfile.ZipFile(path) as z:
            assert z.namelist() == ['submission.json']
            assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
            sub = json.loads(z.read('submission.json'))
        assert len(sub) == 1200 and {it['id'] for it in sub} == set(range(1, 1201))
        assert all(c['doc_id'] in it['relevant_docs'] for it in sub for c in it['relevant_chunks'])
        result.update(queries=1200, entries=1, lzma=True, chunk_doclist_violations=0)
        (STAGE / 'local_verification_c.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)
        return
    if a.action == 'apps':
        env = dict(os.environ, PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
        p = subprocess.run([CLI, 'app', 'list', '--profile', __import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"), '--json'],
                           env=env, capture_output=True, text=True, encoding='utf-8', check=True)
        apps = json.loads(p.stdout)
        assert all(x['state'] == 'stopped' and str(x['tasks']) == '0' for x in apps), apps
        (STAGE / 'modal_apps_final_c.json').write_text(json.dumps(apps, indent=2), encoding='utf-8')
        print(json.dumps({'apps': len(apps), 'all_stopped': True, 'tasks': 0,
                          'volume_kept': True}), flush=True)
        return
    rep = json.loads((STAGE / f'q8b_export_round{r}.json').read_text(encoding='utf-8'))
    for name in rep['files']:
        assert fp(STAGE / 'q8b_input' / name) == rep['fingerprints'][name], name
    if a.action == 'upload':
        for name in rep['files']:
            modal('volume', 'put', '--force', 'r2ai-runner',
                  str(STAGE / 'q8b_input' / name), '/in/vi37/' + name)
    elif a.action == 'get':
        for name in [f'q8b_r{r}_out.json', f'job_report_q8b_r{r}_out.json']:
            modal('volume', 'get', 'r2ai-runner', '/out/vi37/' + name,
                  str(STAGE / 'modal_out' / name))
    else:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
        import vi37
        obj = json.loads((STAGE / f'modal_out/q8b_r{r}_out.json').read_text(encoding='utf-8'))
        expected = json.loads((STAGE / f'q8b_input/cands_q8b_r{r}.json').read_text(encoding='utf-8'))
        vi37.checked_q8b(obj, expected)
        n = sum(map(len, obj.values()))
        assert n == rep['pairs']
        report = json.loads((STAGE / f'modal_out/job_report_q8b_r{r}_out.json').read_text(encoding='utf-8'))
        result = {'pairs': n, 'finite': True, 'gpu_seconds': report['elapsed'],
                  'gpu_usd_est': report['elapsed'] * 1.95 / 3600,
                  'files': {name: fp(STAGE / 'modal_out' / name) for name in
                            [f'q8b_r{r}_out.json', f'job_report_q8b_r{r}_out.json']}}
        (STAGE / f'local_round{r}_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
