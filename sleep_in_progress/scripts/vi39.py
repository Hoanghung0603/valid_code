"""CPU-only vi37 pipeline with reordered LZMA probes; never submits."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import dwmax_diag as dw
import vi23
import vi29 as v29
import vi36
import vi37
from reorder_zip import write_reordered_zip

ROOT = vi37.ROOT
OUT = ROOT / 'data/sub/vi39'
REF = vi37.OUT / 'submission_vi37.json'
LIMIT = 99_000_000


def compare_by_id(items, ref):
    a, b = {str(x['id']): x for x in items}, {str(x['id']): x for x in ref}
    assert len(a) == len(items) and len(b) == len(ref) and set(a) == set(b)
    return {'count': len(a), 'docs': sum(a[q]['relevant_docs'] == b[q]['relevant_docs'] for q in a),
            'chunks': sum(a[q]['relevant_chunks'] == b[q]['relevant_chunks'] for q in a)}


def find_boundary(attempt, start=55, step=32):
    """Bracket then bisect; saturation is exact from the candidate histogram.

    Compression monotonicity is assumed, not proven; all sampled inversions abort.
    """
    measured = {}
    def get(d):
        if d not in measured:
            measured[d] = attempt(d)
            for x, y in zip(sorted(measured), sorted(measured)[1:]):
                if measured[x]['zip_bytes'] > LIMIT >= measured[y]['zip_bytes']:
                    raise RuntimeError(f'nonmonotone fit: {x}, {y}')
        return measured[d]
    first = get(start)
    assert first['zip_bytes'] <= LIMIT, first
    saturation = max(0, max(cap-k for k, cap, n in first['hist_kcap'] if n))
    if saturation <= start:
        return saturation, 'saturated', measured
    lo, hi = start, None
    while lo < saturation:
        d = min(lo + step, saturation)
        if get(d)['zip_bytes'] > LIMIT:
            hi = d
            break
        lo = d
    if hi is None:
        return saturation, 'saturated', measured
    while hi - lo > 1:
        mid = (hi + lo) // 2
        if get(mid)['zip_bytes'] <= LIMIT:
            lo = mid
        else:
            hi = mid
    assert get(lo)['zip_bytes'] <= LIMIT < get(lo + 1)['zip_bytes']
    return lo, 'boundary', measured


def wait_lock(path, label, seconds=7200):
    from contextlib import contextmanager
    @contextmanager
    def locked():
        path.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + seconds
        while True:
            try:
                with path.open('x', encoding='utf-8') as f:
                    f.write(label)
                break
            except FileExistsError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f'lock busy: {path}')
                time.sleep(30)
        try:
            yield
        finally:
            if path.exists() and path.read_text(encoding='utf-8') == label:
                path.unlink()
    return locked()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with wait_lock(ROOT / 'locks/heavy_job.lock', f'vi39 pid={os.getpid()}'):
        build()


def build():
    vi37.require_q8b_complete()
    files = v29.chain_files(vi37.BASE, vi36.FILE_NAMES)
    files['q8b'], files['cands'] = vi37.OUT / 'q8b_v37.json', vi37.OUT / 'cands_v37.json.zst'
    scores = dw.read(vi37.DIAG / 'docwin_max_v37.json')
    ref = dw.read(REF)
    with vi37.chain_docwin(scores):
        v29.run_chain('vi39_sanity', 55, files, 'doc_text', OUT)
        sanity = compare_by_id(dw.read(OUT / 'submission_vi39_sanity.json'), ref)
        dw.write(OUT / 'sanity_report.json', sanity)
        assert sanity == {'count': 1200, 'docs': 1200, 'chunks': 1200}, sanity
        (OUT / 'submission_vi39_sanity.json').unlink()

        cache = OUT / 'attempts_vi39.json'
        tried = {int(k): v for k, v in dw.read(cache).items()} if cache.exists() else {}
        def attempt(delta):
            if delta not in tried:
                rep = v29.run_chain('vi39p', delta, files, 'doc_text', OUT)
                sub = dw.read(OUT / 'submission_vi39p.json')
                result = write_reordered_zip(sub, OUT / 'submission_vi39p.zip')
                result.update({'delta': delta, 'tong_chunk': rep['tong_chunk'],
                               'chunks_per_query': rep['TB_chunk_cau'], 'hist_kcap': rep['hist_kcap']})
                tried[delta] = result
                dw.write(cache, {str(k): v for k, v in sorted(tried.items())})
                print('ATTEMPT', json.dumps({k:v for k,v in result.items() if k != 'hist_kcap'}), flush=True)
            return tried[delta]
        delta, reason, _ = find_boundary(attempt)
        rep = v29.run_chain('vi39', delta, files, 'doc_text', OUT)
        sub = dw.read(OUT / 'submission_vi39.json')
        zp = OUT / 'submission_vi39.zip'
        info = write_reordered_zip(sub, zp)
    assert info['zip_bytes'] <= LIMIT, info
    # check_zip requires identical bytes; materialize the reordered zip's entry.
    ordered = OUT / 'submission_vi39_ordered.json'
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ['submission.json'] and z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        with z.open('submission.json') as src, ordered.open('wb') as dst:
            shutil.copyfileobj(src, dst)
    assert ordered.stat().st_size == info['json_bytes']
    subprocess.run([sys.executable, str(Path(__file__).parent / 'evaluate.py'), 'validate', str(ordered)], check=True)
    code = vi23.check_zip(str(ordered), str(zp), str(OUT / 'report_chain_vi39.json'), 'vi39', max_zip=LIMIT)
    assert code == 0, code
    assert compare_by_id(dw.read(ordered), sub) == {'count': 1200, 'docs': 1200, 'chunks': 1200}
    cmp = compare_by_id(sub, ref)
    assert cmp['count'] == cmp['docs'] == 1200, cmp
    refmap = {str(x['id']): x for x in ref}
    missing = sum(len({(str(c['doc_id']), c['chunk_text']) for c in refmap[str(x['id'])]['relevant_chunks']} -
                      {(str(c['doc_id']), c['chunk_text']) for c in x['relevant_chunks']}) for x in sub)
    bad = set(map(int, dw.read(ROOT / 'data/sub/diag28/bad_ids.json'))) - set(map(int, dw.read(ROOT / 'data/sub/diag30/new_ids.json')))
    corpus = v29.load_corpus_ids(ROOT / 'data/sub/diag19/links_corpus_vi_en_zh.parquet')
    clean = v29.assert_clean(sub, bad, corpus)
    from rerank_run import load_docs
    wanted = {int(c['doc_id']) for it in sub for c in it['relevant_chunks']}
    _, _, textfiles = vi37.export_texts(wanted, OUT / 'verify_text', 'docs_verify')
    docs = load_docs([str(p) for p in textfiles], wanted=wanted)
    verbatim = sum(c['chunk_text'] not in docs[int(c['doc_id'])] for it in sub for c in it['relevant_chunks'])
    assert verbatim == 0, verbatim
    hist = rep['hist_kcap']
    def stats(items, d):
        return {'chunks_per_query': sum(len(x['relevant_chunks']) for x in items)/len(items),
                'docs_with_chunks_per_query': sum(len({str(c['doc_id']) for c in x['relevant_chunks']}) for x in items)/len(items),
                'queries_capped': sum(n for k, cap, n in hist if cap < k+d)}
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    report = {'delta': delta, 'reason': reason, 'attempts': [{k:v for k,v in tried[d].items() if k != 'hist_kcap'} for d in sorted(tried)],
              **info, 'sha256': sha, 'sanity': sanity, 'doc_compare': cmp, 'missing_vi37_chunks': missing,
              'assert_clean': clean, 'verbatim_bad': verbatim, 'validate': 0, 'check_zip': code,
              'stats': {'vi37': stats(ref, 55), 'vi39': stats(sub, delta)},
              'note': 'binary search assumes monotone zip-fit; boundary verified, global maximality not proven'}
    dw.write(OUT / 'build_report_vi39.json', report)
    print('DONE', json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
