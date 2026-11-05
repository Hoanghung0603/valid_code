"""CPU-only vi48: append scored yes>no chunks to vi47, never change its prefix/docs."""
import argparse
from collections import defaultdict
import gc
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

from vi41c import digest, indexed, load_zip
from vi45 import cid, key

LIMIT = 99_000_000


def fingerprint(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return {'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def distribution(values):
    """Linear-interpolated quantiles (same convention as numpy's default)."""
    a = sorted(values)
    if not a:
        return {'count': 0, 'mean': None, 'min': None, 'p10': None,
                'p50': None, 'p90': None, 'p99': None, 'max': None}
    def quantile(p):
        x = (len(a)-1)*p
        lo, hi = math.floor(x), math.ceil(x)
        return a[lo] + (a[hi]-a[lo])*(x-lo)
    return {'count': len(a), 'mean': statistics.mean(a), 'min': a[0],
            'p10': quantile(.1), 'p50': quantile(.5), 'p90': quantile(.9),
            'p99': quantile(.99), 'max': a[-1]}


def select_row(base, scored):
    """scored = iterable of (candidate, finite logit). Stable cid tie-break."""
    docs = set(map(str, base['relevant_docs']))
    seen = {key(c) for c in base['relevant_chunks']}
    if len(seen) != len(base['relevant_chunks']):
        raise ValueError('duplicate baseline chunk')
    if not {d for d, _ in seen} <= docs:
        raise ValueError('baseline chunk outside docs')
    extra, excluded = [], []
    used_cids = set()
    for c, logit in scored:
        if not math.isfinite(logit):
            raise ValueError('nonfinite score')
        identity = cid(base['id'], c)
        if c['cid'] != identity or identity in used_cids:
            raise ValueError('invalid or duplicate candidate cid')
        used_cids.add(identity)
        if logit <= 0 or key(c) in seen:
            continue
        if str(c['doc_id']) not in docs:
            excluded.append({'doc_id': c['doc_id'], 'cid': identity, 'logit8c': logit})
            continue
        seen.add(key(c))
        extra.append({'doc_id': c['doc_id'], 'chunk_text': c['chunk_text'],
                      'cid': identity, 'logit8c': logit})
    extra.sort(key=lambda c: (-c['logit8c'], c['cid']))
    return extra, excluded


def append(base, additions, removed=frozenset()):
    out = []
    for row in base:
        extra = [dict(doc_id=c['doc_id'], chunk_text=c['chunk_text'])
                 for c in additions[str(row['id'])] if c['cid'] not in removed]
        out.append({**row, 'relevant_chunks': row['relevant_chunks'] + extra})
    return out


def fit_budget(base, additions, measure, limit=LIMIT):
    """Only budget repair, not tuning: drop exactly one global lowest logit at a time.

    Actual reordered ZIP is measured after each removal; no monotonicity assumption.
    Equal logits: descending cid is dropped first, preserving ascending cid on ties.
    """
    removed, attempts = set(), []
    lowest = sorted((c for rows in additions.values() for c in rows),
                    key=lambda c: (c['logit8c'], -int(c['cid'], 16)))
    while True:
        sub = append(base, additions, removed)
        info = measure(sub)
        attempts.append({'removed': len(removed), **info})
        if info['zip_bytes'] <= limit:
            return sub, removed, attempts
        if len(removed) == len(lowest):
            raise ValueError('baseline alone exceeds budget')
        removed.add(lowest[len(removed)]['cid'])


def check_relations(sub, base, additions, removed=frozenset()):
    actual, ref = indexed(sub), indexed(base)
    assert set(actual) == set(ref)
    for q, row in actual.items():
        old = ref[q]
        assert {k: v for k, v in row.items() if k != 'relevant_chunks'} == {
            k: v for k, v in old.items() if k != 'relevant_chunks'}
        chunks = row['relevant_chunks']
        assert chunks[:len(old['relevant_chunks'])] == old['relevant_chunks']
        wanted = [dict(doc_id=c['doc_id'], chunk_text=c['chunk_text'])
                  for c in additions[str(q)] if c['cid'] not in removed]
        assert chunks[len(old['relevant_chunks']):] == wanted
        pairs = [key(c) for c in chunks]
        assert len(pairs) == len(set(pairs))
        assert {d for d, _ in pairs} <= set(map(str, row['relevant_docs']))
    return {'vi45_literal_prefix': len(ref), 'docs_equal_vi47': len(ref),
            'other_fields_equal_vi47': len(ref), 'append_order_exact': len(ref),
            'duplicate_chunks': 0, 'chunks_outside_docs': 0}


def verify_verbatim(sub, files):
    """Verify all final chunks by streaming the frozen vi45 full-text export."""
    import zstandard as zstd
    pending = defaultdict(set)
    count = 0
    for row in sub:
        for c in row['relevant_chunks']:
            pending[int(c['doc_id'])].add(c['chunk_text'])
            count += 1
    unique = sum(map(len, pending.values()))
    for p in files:
        with zstd.open(p, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                texts = pending.pop(int(r['id']), None)
                if texts is not None:
                    assert all(t in r['text'] for t in texts), ('not verbatim', r['id'])
    assert not pending, ('missing export docs', len(pending))
    return {'verbatim_bad': 0, 'verbatim_all_final_chunks': count,
            'verbatim_unique_doc_text': unique}


def build(args):
    import zstandard as zstd
    from reorder_zip import write_reordered_zip
    start = time.monotonic()
    source, out = args.vi45_dir, args.out
    zp, jp = out/'submission_vi48.zip', out/'submission_vi48.json'
    if zp.exists() or jp.exists() or (out/'build_report_vi48.json').exists():
        raise FileExistsError('vi48 output exists; refusing overwrite')
    input_paths = [args.vi47, args.vi45, args.queries, source/'bad_ids.json',
                   source/'new_ids.json', source/'links_corpus_vi_en_zh.parquet',
                   source/'prepare_vi44_report.json', source/'score_transfer_manifest.json',
                   source/'preverify_report.json']
    before = {str(p): fingerprint(p) for p in input_paths}
    base = sorted(load_zip(args.vi47), key=lambda r: r['id'])
    expected = {json.loads(line)['id'] for line in args.queries.read_text(encoding='utf-8').splitlines() if line.strip()}
    assert len(base) == len(expected) == 1200
    assert set(indexed(base)) == expected == set(range(1, 1201))
    base_chunks = {r['id']: digest(r['relevant_chunks']) for r in base}
    vi45 = load_zip(args.vi45)
    assert set(indexed(vi45)) == expected
    assert all(digest(r['relevant_chunks']) == base_chunks[r['id']] for r in vi45)
    del vi45
    gc.collect()
    print('BASE_PREFIX_PASS', flush=True)
    manifest = read(source/'prepare_vi44_report.json')
    score_manifest = read(source/'score_transfer_manifest.json')['files']
    scorefiles = sorted((source/'outputs').rglob('*.logits.json'))
    assert len(scorefiles) == len(score_manifest) == 20
    assert {p.name: fingerprint(p) for p in scorefiles} == score_manifest
    scores = {}
    for p in scorefiles:
        for r in read(p):
            identity = r['cid']
            assert identity not in scores and math.isfinite(r['logit8c'])
            scores[identity] = (str(r['qid']), str(r['doc_id']), float(r['logit8c']))
    assert len(scores) == manifest['scored_candidates'] == 317601
    # Read the already-scored input, not the unscored candidate tail or bench.
    byquery = defaultdict(list)
    matched = 0
    for name, meta in sorted(manifest['files'].items()):
        p = source/'modal_input'/name
        assert fingerprint(p) == meta
        before[str(p)] = meta
        n = 0
        with zstd.open(p, 'rt', encoding='utf-8') as f:
            for line in f:
                r = json.loads(line)
                q, d, logit = scores.pop(r['cid'])
                assert (q, d) == (str(r['qid']), str(r['doc_id']))
                c = dict(doc_id=r['doc_id'], chunk_text=r['text'], cid=r['cid'])
                assert cid(q, c) == r['cid']
                byquery[q].append((c, logit))
                matched += 1
                n += 1
        assert n == manifest['counts'][name]
    assert not scores and matched == 317601
    for p in scorefiles:
        before[str(p)] = score_manifest[p.name]
    additions, excluded, baseline_positive, baseline_scored = {}, [], 0, 0
    for row in base:
        q = str(row['id'])
        scored = byquery.pop(q)
        logits = {c['cid']: s for c, s in scored}
        for c in row['relevant_chunks']:
            s = logits.get(cid(q, c))
            baseline_scored += s is not None
            baseline_positive += s is not None and s > 0
        additions[q], rejected = select_row(row, scored)
        excluded.extend({'qid': q, **r} for r in rejected)
    assert not byquery
    del scored, logits, scores, byquery
    gc.collect()
    pre_count = sum(map(len, additions.values()))
    write(out/'selection_vi48.json', {'eligible_additions': pre_count,
          'excluded_positive_nonbase_chunks_doc_not_vi47': len(excluded),
          'excluded': excluded,
          'additions': {q: [{k: v for k, v in c.items() if k != 'chunk_text'} for c in cs]
                        for q, cs in additions.items()}})
    print('SELECTED '+json.dumps({'additions': pre_count, 'excluded': len(excluded)}), flush=True)
    sub, removed, attempts = fit_budget(base, additions, lambda rows: write_reordered_zip(rows, zp))
    relations = check_relations(sub, base, additions, removed)
    retained = [c for cs in additions.values() for c in cs if c['cid'] not in removed]
    rejected_budget = [c for cs in additions.values() for c in cs if c['cid'] in removed]
    counts = [sum(c['cid'] not in removed for c in additions[str(r['id'])]) for r in base]
    offline = {'added_per_query': distribution(counts), 'zero_added_queries': counts.count(0),
        'added_logit8c': distribution([c['logit8c'] for c in retained]),
        'final_chunks_per_query': distribution([len(r['relevant_chunks']) for r in sub]),
        'vi45_chunk_count': sum(len(r['relevant_chunks']) for r in base),
        'vi45_chunks_scored': baseline_scored, 'vi45_chunks_logit8c_positive': baseline_positive,
        'vi45_fraction_logit8c_positive': baseline_positive/sum(len(r['relevant_chunks']) for r in base),
        'eligible_additions_before_budget': pre_count, 'removed_budget': len(removed),
        'min_retained_logit8c': min((c['logit8c'] for c in retained), default=None),
        'max_removed_logit8c': max((c['logit8c'] for c in rejected_budget), default=None),
        'excluded_positive_nonbase_chunks_doc_not_vi47': len(excluded),
        'excluded_query_doc_pairs': len({(r['qid'], str(r['doc_id'])) for r in excluded}),
        'per_query': [{'id': r['id'], 'added': n, 'final_chunks': len(s['relevant_chunks'])}
                      for r, s, n in zip(base, sub, counts)]}
    write(out/'offline_vi48.json', offline)
    preverify = read(source/'preverify_report.json')['files']
    texts = []
    for remote, meta in preverify.items():
        p = source/remote.replace('\\', '/').split('/')[-1]
        assert fingerprint(p) == meta
        before[str(p)] = meta
        texts.append(p)
    verbatim = verify_verbatim(sub, texts)
    from vi29 import assert_clean, load_corpus_ids
    bad = set(map(int, read(source/'bad_ids.json'))) - set(map(int, read(source/'new_ids.json')))
    corpus = load_corpus_ids(source/'links_corpus_vi_en_zh.parquet')
    clean = assert_clean(sub, bad, corpus)
    original_hashes = {r['id']: digest(r) for r in sub}
    del sub, base, additions, retained, rejected_budget, corpus
    gc.collect()
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ['submission.json']
        assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        assert z.testzip() is None
        with z.open('submission.json') as src, jp.open('xb') as dst:
            shutil.copyfileobj(src, dst)
    final = load_zip(zp)
    assert set(indexed(final)) == expected
    assert {r['id']: digest(r) for r in final} == original_hashes
    del final
    gc.collect()
    subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate.py')),
                    'validate', str(jp)], check=True)
    # Isolate the real check_zip's large byte buffers from this process's heap.
    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'check-zip',
                    '--out', str(out)], check=True)
    assert all(fingerprint(p) == meta for p, meta in before.items())
    assert zp.stat().st_size <= LIMIT
    report = {**attempts[-1], **fingerprint(zp), **relations, **verbatim,
        'validate': 0, 'check_zip': 0, 'assert_clean': clean, 'crc': 0,
        'entry': 'submission.json', 'compression': 'LZMA', 'ids': 1200,
        'zip_content_equal_before_reorder': 1200, 'vi47_chunks_equal_vi45': 1200,
        'added_verbatim_count': pre_count-len(removed), 'scored_pairs_exact': matched,
        'budget_attempts': attempts, 'removed_cids': sorted(removed),
        'rule': 'vi47 prefix/docs unchanged; append scored nonbase logit8c>0 within vi47 docs; descending logit, ascending cid ties',
        'input_fingerprints': before, 'inputs_unchanged': True,
        'cpu_only': True, 'usd': 0, 'seconds': time.monotonic()-start}
    write(out/'build_report_vi48.json', report)
    print('DONE '+json.dumps({k: v for k, v in report.items() if k != 'input_fingerprints'}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['build', 'check-zip'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--vi45-dir', type=Path)
    ap.add_argument('--vi45', type=Path)
    ap.add_argument('--vi47', type=Path)
    ap.add_argument('--queries', type=Path)
    args = ap.parse_args()
    if args.command == 'check-zip':
        from vi23 import check_zip
        code = check_zip(str(args.out/'submission_vi48.json'), str(args.out/'submission_vi48.zip'),
                         str(args.out/'check_zip.json'), 'vi48', max_zip=LIMIT)
        raise SystemExit(code)
    if any(getattr(args, x) is None for x in ['vi45_dir', 'vi45', 'vi47', 'queries']):
        ap.error('build needs --vi45-dir, --vi45, --vi47, --queries')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out/'build.lock'
    with lock.open('x', encoding='utf-8') as f:
        import os
        f.write(f'vi48 CPU local pid={os.getpid()}')
    try:
        build(args)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
