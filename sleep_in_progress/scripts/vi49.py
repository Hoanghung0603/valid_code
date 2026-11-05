"""CPU-only vi49: fixed global 8B logit<=0 prefix, bounded by reordered ZIP bytes."""
import argparse
from collections import defaultdict
import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from vi48 import (fingerprint, read, write, distribution, verify_verbatim,
                  digest, indexed, load_zip, cid, key)

LIMIT = 98_500_000


def select_row(base, scored):
    docs = set(map(str, base['relevant_docs']))
    seen = {key(c) for c in base['relevant_chunks']}
    if len(seen) != len(base['relevant_chunks']):
        raise ValueError('duplicate baseline chunk')
    if not {d for d, _ in seen} <= docs:
        raise ValueError('baseline chunk outside docs')
    used, eligible, excluded = set(), [], 0
    # Sort before dedup, so even duplicate text has a deterministic winner.
    checked = []
    for c, score in scored:
        identity = cid(base['id'], c)
        if not math.isfinite(score):
            raise ValueError('nonfinite score')
        if c['cid'] != identity or identity in used:
            raise ValueError('invalid or duplicate candidate cid')
        used.add(identity)
        checked.append((c, score))
    for c, score in sorted(checked, key=lambda x: (-x[1], x[0]['cid'])):
        if score > 0 or key(c) in seen:
            continue
        if str(c['doc_id']) not in docs:
            excluded += 1
            continue
        seen.add(key(c))
        eligible.append({**c, 'qid': str(base['id']), 'logit8c': score})
    return eligible, excluded


def global_order(additions):
    return sorted((c for cs in additions.values() for c in cs),
                  key=lambda c: (-c['logit8c'], c['cid']))


def append_prefix(base, ranked, count):
    additions = defaultdict(list)
    for c in ranked[:count]:
        additions[c['qid']].append({'doc_id': c['doc_id'], 'chunk_text': c['chunk_text']})
    return [{**r, 'relevant_chunks': r['relevant_chunks'] + additions[str(r['id'])]}
            for r in base]


def check_relations(sub, base, ranked, count):
    expected = indexed(append_prefix(base, ranked, count))
    actual = indexed(sub)
    assert set(actual) == set(expected)
    for q, r in actual.items():
        assert r == expected[q], ('prefix/docs/exact suffix', q)
        pairs = [key(c) for c in r['relevant_chunks']]
        assert len(pairs) == len(set(pairs))
        assert {d for d, _ in pairs} <= set(map(str, r['relevant_docs']))
    return {'vi48_literal_prefix': len(base), 'docs_equal_vi48': len(base),
            'other_fields_equal_vi48': len(base), 'append_order_exact': len(base),
            'duplicate_chunks': 0, 'chunks_outside_docs': 0}


def fit_budget(base, ranked, measure, base_bytes, limit=LIMIT, max_attempts=10):
    """Byte-secant estimates, then measure adjacent boundary; never skip a candidate.

    Reorder/LZMA size is not mathematically monotone: the certificate is the actual
    chosen prefix fitting and its immediate next prefix overflowing, not a claim
    about every larger prefix. Each trial uses the real reorder_zip writer.
    """
    if base_bytes > limit:
        raise ValueError('baseline alone exceeds budget')
    total = len(ranked)
    if not total:
        info = measure(base, 0)
        return base, 0, [{'count': 0, **info}]
    # vi48's measured incremental bytes/chunk is only an initial estimate.
    count = min(total, max(1, int((limit-base_bytes)/327)))
    measured = {0: base_bytes}
    attempts = []
    for _ in range(max_attempts):
        sub = append_prefix(base, ranked, count)
        info = measure(sub, count)
        size = info['zip_bytes']
        measured[count] = size
        attempts.append({'count': count, **info})
        if size <= limit and (count == total or
                              measured.get(count+1, 0) > limit):
            return sub, count, attempts
        feasible = max(n for n, b in measured.items() if b <= limit)
        overflow = min((n for n, b in measured.items() if b > limit and n > feasible),
                       default=None)
        if overflow == feasible+1:
            # Restore the certified feasible ZIP (measure may overwrite its path).
            sub = append_prefix(base, ranked, feasible)
            info = measure(sub, feasible)
            assert info['zip_bytes'] == measured[feasible]
            attempts.append({'count': feasible, 'restore': True, **info})
            return sub, feasible, attempts
        if overflow is None:
            slope = max((size-base_bytes)/max(count, 1), 1.)
            nxt = count + max(1, int((limit-size)/slope))
            nxt = min(total, nxt)
        else:
            low_size, high_size = measured[feasible], measured[overflow]
            fraction = (limit-low_size)/max(high_size-low_size, 1)
            nxt = feasible + max(1, int((overflow-feasible)*fraction))
            nxt = min(overflow-1, nxt)
        if nxt in measured:
            nxt = feasible+1 if overflow is None else (feasible+overflow)//2
        count = nxt
        del sub
        gc.collect()
    raise RuntimeError('budget boundary not certified within allowed real builds')


def build(args):
    import zstandard as zstd
    from reorder_zip import write_reordered_zip
    start = time.monotonic()
    source, out = args.vi45_dir, args.out
    zp, jp = out/'submission_vi49.zip', out/'submission_vi49.json'
    if zp.exists() or jp.exists() or (out/'build_report_vi49.json').exists():
        raise FileExistsError('vi49 output exists; refusing overwrite')
    paths = [args.vi48, args.queries, source/'bad_ids.json', source/'new_ids.json',
             source/'links_corpus_vi_en_zh.parquet', source/'prepare_vi44_report.json',
             source/'score_transfer_manifest.json', source/'preverify_report.json']
    before = {str(p): fingerprint(p) for p in paths}
    base = sorted(load_zip(args.vi48), key=lambda r: r['id'])
    expected = {json.loads(s)['id'] for s in args.queries.read_text(encoding='utf-8').splitlines() if s.strip()}
    assert len(base) == len(expected) == 1200
    assert set(indexed(base)) == expected == set(range(1, 1201))
    manifest = read(source/'prepare_vi44_report.json')
    score_manifest = read(source/'score_transfer_manifest.json')['files']
    scorefiles = sorted((source/'outputs').rglob('*.logits.json'))
    assert len(scorefiles) == len(score_manifest) == 20
    assert {p.name: fingerprint(p) for p in scorefiles} == score_manifest
    scores = {}
    for p in scorefiles:
        for r in read(p):
            assert r['cid'] not in scores and math.isfinite(r['logit8c'])
            scores[r['cid']] = (str(r['qid']), str(r['doc_id']), float(r['logit8c']))
        before[str(p)] = score_manifest[p.name]
    assert len(scores) == manifest['scored_candidates'] == 317601
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
                q, d, score = scores.pop(r['cid'])
                assert (q, d) == (str(r['qid']), str(r['doc_id']))
                c = dict(doc_id=r['doc_id'], chunk_text=r['text'], cid=r['cid'])
                assert cid(q, c) == r['cid']
                byquery[q].append((c, score))
                n += 1
        assert n == manifest['counts'][name]
        matched += n
    assert not scores and matched == 317601
    additions, excluded = {}, 0
    for r in base:
        q = str(r['id'])
        additions[q], rejected = select_row(r, byquery.pop(q))
        excluded += rejected
    assert not byquery
    ranked = global_order(additions)
    del additions, byquery, scores
    gc.collect()
    write(out/'selection_vi49.json', {'eligible_additions': len(ranked),
          'excluded_nonpositive_outside_docs': excluded,
          'global_order': [{k: v for k, v in c.items() if k != 'chunk_text'} for c in ranked]})
    print('SELECTED '+str(len(ranked)), flush=True)
    def measure(rows, count):
        t = time.monotonic()
        info = write_reordered_zip(rows, zp)
        info['seconds'] = time.monotonic()-t
        with (out/'budget_trials.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'count': count, **info})+'\n')
        print('MEASURE '+json.dumps({'count': count, **info}), flush=True)
        return info
    sub, count, attempts = fit_budget(base, ranked, measure,
                                     before[str(args.vi48)]['bytes'])
    relations = check_relations(sub, base, ranked, count)
    counts = defaultdict(int)
    for c in ranked[:count]:
        counts[c['qid']] += 1
    nums = [counts[str(r['id'])] for r in base]
    offline = {'added_total': count, 'added_per_query': distribution(nums),
        'zero_added_queries': nums.count(0),
        'added_logit8c': distribution([c['logit8c'] for c in ranked[:count]]),
        'final_chunks_per_query': distribution([len(r['relevant_chunks']) for r in sub]),
        'eligible_additions': len(ranked), 'baseline_chunks': sum(len(r['relevant_chunks']) for r in base),
        'cutoff': {k: v for k, v in ranked[count-1].items() if k != 'chunk_text'} if count else None,
        'next_candidate': {k: v for k, v in ranked[count].items() if k != 'chunk_text'} if count < len(ranked) else None,
        'per_query': [{'id': r['id'], 'added': n, 'final_chunks': len(s['relevant_chunks'])}
                      for r, s, n in zip(base, sub, nums)]}
    write(out/'offline_vi49.json', offline)
    texts = []
    for remote, meta in read(source/'preverify_report.json')['files'].items():
        p = source/remote.replace('\\', '/').split('/')[-1]
        assert fingerprint(p) == meta
        before[str(p)] = meta
        texts.append(p)
    verbatim = verify_verbatim(sub, texts)
    from vi29 import assert_clean, load_corpus_ids
    bad = set(map(int, read(source/'bad_ids.json'))) - set(map(int, read(source/'new_ids.json')))
    clean = assert_clean(sub, bad, load_corpus_ids(source/'links_corpus_vi_en_zh.parquet'))
    original = {r['id']: digest(r) for r in sub}
    del sub, base, ranked
    gc.collect()
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ['submission.json']
        assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        assert z.testzip() is None
        with z.open('submission.json') as src, jp.open('xb') as dst:
            shutil.copyfileobj(src, dst)
    final = load_zip(zp)
    assert set(indexed(final)) == expected
    assert {r['id']: digest(r) for r in final} == original
    del final
    gc.collect()
    subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate.py')),
                    'validate', str(jp)], check=True)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), 'check-zip',
                    '--out', str(out)], check=True)
    assert all(fingerprint(p) == meta for p, meta in before.items())
    assert zp.stat().st_size <= LIMIT
    report = {**attempts[-1], **fingerprint(zp), **relations, **verbatim,
        'validate': 0, 'check_zip': 0, 'assert_clean': clean, 'crc': 0,
        'entry': 'submission.json', 'compression': 'LZMA', 'ids': 1200,
        'zip_content_equal_before_reorder': 1200, 'added_verbatim_count': count,
        'scored_pairs_exact': matched, 'budget_attempts': attempts,
        'next_prefix_overflow_measured': count == offline['eligible_additions'] or any(
            a['count'] == count+1 and a['zip_bytes'] > LIMIT for a in attempts),
        'rule': 'vi48 exact prefix/docs; global scored nonbase logit8c<=0 in docs; descending logit then ascending CID; maximal adjacent fitting prefix',
        'input_fingerprints': before, 'inputs_unchanged': True,
        'cpu_only': True, 'usd': 0, 'seconds': time.monotonic()-start}
    write(out/'build_report_vi49.json', report)
    jp.unlink()
    print('DONE '+json.dumps({k: v for k, v in report.items() if k != 'input_fingerprints'}), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command', choices=['build', 'check-zip'])
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--vi45-dir', type=Path)
    ap.add_argument('--vi48', type=Path)
    ap.add_argument('--queries', type=Path)
    args = ap.parse_args()
    if args.command == 'check-zip':
        from vi23 import check_zip
        raise SystemExit(check_zip(str(args.out/'submission_vi49.json'),
            str(args.out/'submission_vi49.zip'), str(args.out/'check_zip.json'), 'vi49', max_zip=LIMIT))
    if any(getattr(args, x) is None for x in ['vi45_dir', 'vi48', 'queries']):
        ap.error('build needs --vi45-dir, --vi48, --queries')
    args.out.mkdir(parents=True, exist_ok=True)
    lock = args.out/'build.lock'
    with lock.open('x', encoding='utf-8') as f:
        import os
        f.write(f'vi49 CPU local pid={os.getpid()}')
    try:
        build(args)
    finally:
        lock.unlink()


if __name__ == '__main__':
    main()
