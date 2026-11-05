"""Bounded Modal launcher using existing modal_runner jobs, explicit profile and <=3 GPUs.

No input upload, no submission. Abort on billing/quota errors; reports elapsed GPU estimates.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys

CLI = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe')
REPO = Path(__file__).resolve().parents[1]
STAGING = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi37')


def job(stage, cands, out):
    env = dict(os.environ, PYTHONIOENCODING='utf-8')
    cmd = [str(CLI), 'run', '--profile', __import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"), 'deploy/modal_runner.py',
           '--stage', stage, '--tag', 'vi37']
    if cands:
        cmd += ['--cands', cands, '--out', out]
    log = STAGING / f'modal_{stage}_{out or "bench_bge"}.log'
    with log.open('w', encoding='utf-8') as f:
        p = subprocess.run(cmd, cwd=REPO, env=env, stdout=f, stderr=subprocess.STDOUT)
    text = log.read_text(encoding='utf-8')
    errors = ['ConflictError', 'is disabled', 'quota', 'billing', 'insufficient credits']
    if p.returncode or any(s.lower() in text.lower() for s in errors):
        raise RuntimeError(text[-6000:])
    print('DONE', stage, out or 'bench_bge', flush=True)
    return str(log)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('phase', choices=['bench', 'docwin', 'q8b-bench', 'q8b'])
    a = ap.parse_args()
    STAGING.mkdir(parents=True, exist_ok=True)
    if a.phase == 'bench':
        job('bench_bge', '', '')
    elif a.phase == 'q8b-bench':
        job('q8b', 'cands_q8b_bench.json', 'q8b_bench_out.json')
    else:
        if a.phase == 'q8b':
            sys.path.insert(0, str(REPO / 'scripts'))
            import dwmax_diag as dw
            bench = dw.read(STAGING / 'q8b_input/bench_q8b_old.json')
            got = {q: {str(h[0]): h[2] for h in hits} for q, hits in
                   dw.read(STAGING / 'modal_out/q8b_bench_out.json').items()}
            pairs = [(s, got[q][str(d)]) for q, d, s in bench]
            report = dw.overlap_report(pairs)
            assert len(pairs) == 200 and report['pearson'] >= .999, report
            print('BENCH_GATE', json.dumps(report), flush=True)
        tasks = ([('docwin', f'cands_docwin_shard{i}.json', f'docwin_shard{i}.json.zst') for i in range(3)]
                  if a.phase == 'docwin' else
                  [('q8b', f'cands_q8b_shard{i}.json', f'q8b_shard{i}_out.json') for i in range(2)])
        with ThreadPoolExecutor(max_workers=3) as ex:
            futures = [ex.submit(job, *t) for t in tasks]
            for f in as_completed(futures):
                f.result()


if __name__ == '__main__':
    main()
