"""vi37 with docwin_max x1 in chunk zd only; CPU, never submit."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import zipfile

import fuse3
import vi23
import vi29 as v29
import vi36
import vi37 as v37

ROOT = v37.ROOT
OUT = ROOT / 'data/sub/vi37z'
REFERENCE = ROOT / 'data/sub/vi37'
SCORES = v37.DIAG / 'docwin_max_v37.json'
SCORES_SHA = '0721a7c3d6d3eb3e97df02feebacbe02f456a32776837c13aeb96e288db47b0c'


@contextmanager
def chain_docwin_zd(scores, enabled=True):
    """Reuse vi37's fhyde hook; add the identical source only to rrf3d2."""
    with v37.chain_docwin(scores):
        original = fuse3.fuse3
        def fuse(pool, q8b, dense, weights=(1., 1., 1.), k=60, bm25=None, extras=None):
            ex = list(extras or [])
            if enabled and len(weights) > 3 and weights[3] == 0:
                ex.append((v37.docwin_hits(pool, scores), 1.))
            return original(pool, q8b, dense, weights, k, bm25, ex)
        fuse3.fuse3 = fuse
        try:
            yield
        finally:
            fuse3.fuse3 = original


def checked_scores():
    with SCORES.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else None
    if digest is None:  # Python 3.10 on b660
        h = hashlib.sha256()
        with SCORES.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        digest = h.hexdigest()
    assert digest == SCORES_SHA, f'docwin SHA mismatch: {digest}'
    return v37.dw.read(SCORES)


def files():
    result = v29.chain_files(v37.BASE, vi36.FILE_NAMES)
    result['q8b'] = REFERENCE / 'q8b_v37.json'
    result['cands'] = REFERENCE / 'cands_v37.json.zst'
    return result


def sanity(scores):
    v29.run_chain('vi37z_sanity', 55, files(), 'doc_text', OUT)
    path = OUT / 'submission_vi37z_sanity.json'
    cmp = vi23.compare_subs(v37.dw.read(path), str(REFERENCE / 'submission_vi37.json'))
    v29.require_sanity(cmp)
    cmp['byte_identical'] = path.read_bytes() == (REFERENCE / 'submission_vi37.json').read_bytes()
    assert cmp['byte_identical'], 'sanity JSON bytes differ'
    v37.dw.write(OUT / 'sanity_report.json', cmp)
    print('SANITY', json.dumps(cmp), flush=True)


def build(scores):
    v37.require_q8b_complete()  # checks the existing vi37 selection, never requests new scoring
    oldout = v29.OUT
    v29.OUT = OUT
    try:
        with chain_docwin_zd(scores):
            delta, attempts = v29.find_delta(files(), 'doc_text', 99000000, 56, 'vi37zp')
            check = v29.chain_zip_check('vi37z', delta, 99000000, files(), 'doc_text', validate=True)
    finally:
        v29.OUT = oldout
    assert check['code'] == 0, check
    sub = v37.dw.read(OUT / 'submission_vi37z.json')
    ref = v37.dw.read(REFERENCE / 'submission_vi37.json')
    sm, rm = {str(x['id']): x for x in sub}, {str(x['id']): x for x in ref}
    assert len(sm) == len(rm) == 1200 and set(sm) == set(rm)
    assert all(sm[q]['relevant_docs'] == rm[q]['relevant_docs'] for q in sm), 'doclist changed'
    excl = set(map(int, v37.dw.read(ROOT / 'data/sub/diag28/bad_ids.json'))) - set(
        map(int, v37.dw.read(ROOT / 'data/sub/diag30/new_ids.json')))
    corpus = v29.load_corpus_ids(ROOT / 'data/sub/diag19/links_corpus_vi_en_zh.parquet')
    clean = v29.assert_clean(sub, excl, corpus)
    from rerank_run import load_docs
    wanted = {int(c['doc_id']) for it in sub for c in it['relevant_chunks']}
    _, _, textfiles = v37.export_texts(wanted, OUT / 'verify_text', 'docs_verify')
    docs = load_docs([str(p) for p in textfiles], wanted=wanted)
    verbatim_bad = sum(c['chunk_text'] not in docs[int(c['doc_id'])]
                       for it in sub for c in it['relevant_chunks'])
    assert verbatim_bad == 0, verbatim_bad
    zp = OUT / 'submission_vi37z.zip'
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ['submission.json'] and z.infolist()[0].compress_type == zipfile.ZIP_LZMA
    assert zp.stat().st_size <= 99000000
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    report = {'delta': delta, 'attempts': attempts, 'check_zip': check, 'validate_exit': 0,
              'assert_clean': clean, 'verbatim_bad': verbatim_bad, 'docs_order_equal': 1200,
              'zip_bytes': zp.stat().st_size, 'sha256': sha, 'docwin_sha256': SCORES_SHA}
    v37.dw.write(OUT / 'build_report_vi37z.json', report)
    stats(sub, ref, scores, delta)
    print('BUILD', json.dumps(report), flush=True)


def stats(sub, ref, scores, delta):
    sm, rm = {str(x['id']): x for x in sub}, {str(x['id']): x for x in ref}
    rows = {}
    data = v37.inputs()
    q8b = v37.dw.read(REFERENCE / 'q8b_v37.json')
    cands = v37.dw.read(REFERENCE / 'cands_v37.json.zst')
    kq = {str(it['id']): len(it['relevant_chunks']) for it in v37.dw.read(v29.REF_K)}
    for name, items, d, enabled in [('vi37', ref, 55, False), ('vi37z', sub, delta, True)]:
        lists = {str(it['id']): [str(x) for x in it['relevant_docs']] for it in items}
        with chain_docwin_zd(scores, enabled):
            rrf = fuse3.fuse3(data['pool'], q8b, data['dense'], (1., 1., 2., 0.), 60)
        rows[name] = {'chunks_per_query': sum(len(it['relevant_chunks']) for it in items)/1200,
                      'chunk_source_docs_per_query': sum(len({str(c['doc_id']) for c in it['relevant_chunks']}) for it in items)/1200,
                      'delta': d, 'queries_capped_by_Cq': v37.count_capped(cands, rrf, lists, kq, d)}
    pairs = lambda it: {(str(c['doc_id']), c['chunk_text']) for c in it['relevant_chunks']}
    overlap = jaccard = unchanged = 0
    for q in sm:
        a, b = pairs(sm[q]), pairs(rm[q])
        overlap += len(a & b)
        jaccard += len(a & b)/len(a | b) if a | b else 1.
        unchanged += a == b
    report = {'rows': rows, 'chunk_exact_overlap_mean': overlap/1200,
              'chunk_jaccard_macro': jaccard/1200, 'queries_chunk_set_unchanged': unchanged,
              'queries_chunk_order_unchanged': sum(sm[q]['relevant_chunks'] == rm[q]['relevant_chunks'] for q in sm)}
    v37.dw.write(OUT / 'stats_report_vi37z.json', report)
    print('STATS', json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['sanity', 'build', 'all'])
    ap.add_argument('--no-docwin-zd', action='store_true')
    a = ap.parse_args()
    if a.stage == 'sanity' and not a.no_docwin_zd:
        ap.error('sanity requires --no-docwin-zd')
    if a.stage != 'sanity' and a.no_docwin_zd:
        ap.error('--no-docwin-zd is only for sanity')
    OUT.mkdir(parents=True, exist_ok=True)
    with v37.dw.owned_lock(ROOT / 'locks/heavy_job.lock', f'vi37z {a.stage} pid={os.getpid()}'):
        scores = checked_scores()
        if a.stage in ('sanity', 'all'):
            with chain_docwin_zd(scores, enabled=False):
                sanity(scores)
        if a.stage in ('build', 'all'):
            assert v37.dw.read(OUT / 'sanity_report.json')['byte_identical']
            build(scores)
    print('DONE', a.stage, flush=True)


if __name__ == '__main__':
    main()
