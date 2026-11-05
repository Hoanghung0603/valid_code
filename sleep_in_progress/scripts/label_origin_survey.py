"""CPU/cache-only label-origin diagnostic. Never reads gold, crawl, or changes vi*.

Run --inspect first; normal run writes only data/sub/label_origin_survey/.
Missing top-k hit is right-censored, NOT a measured rank or nonrelevance label.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time


KS = (10, 30, 50, 100, 200)
SPECS = {
    'bm25': 'vi36/cache_vi36_k300.json',
    'bm25_old': 'cache_vi_k300.json',
    'bm25c_syl': 'bm25c/cache_bm25c_syl.json',
    'bm25c_pyvi': 'bm25c/cache_bm25c_pyvi.json',
    'dense': 'vi36/in1/dense_v36.json',
    'hyde': 'vi36/in1/hyde_v36.json',
    'bge': 'vi36/in1/pool_v36.json',
    'q8b': 'vi37/q8b_v37.json',
    'fhyde': 'vi37/cache_fhyde_v37.json',
}


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def ranked(rows):
    """Score order, stable ties, first occurrence of an ID wins."""
    result = {}
    for row in sorted(rows, key=lambda h: -float(h[2])):
        result.setdefault(int(row[0]), len(result) + 1)
    return result


def marginal_precision(n0, p0, n1, p1):
    """Exact only for nested lists with query-constant sizes; otherwise proxy."""
    if n1 == n0:
        return None
    return (n1 * p1 - n0 * p0) / (n1 - n0)


def summarize(ranks, cap):
    known = [r for r in ranks if r is not None]
    n = len(ranks)
    return {'n': n, 'observed': len(known),
            'observed_mean': statistics.mean(known) if known else None,
            'observed_median': statistics.median(known) if known else None,
            'clipped_mean': statistics.mean(r if r is not None else cap + 1 for r in ranks) if n else None,
            'topk_fraction': {str(k): sum(r is not None and r <= k for r in ranks) / n if n else None
                              for k in KS}}


def paired_bootstrap(values, repetitions=2000):
    """Cluster on affected queries; values=(query delta sum, pair count)."""
    if not values:
        return None
    rng = random.Random(7611)
    estimates = []
    for _ in range(repetitions):
        sample = rng.choices(values, k=len(values))
        estimates.append(sum(x[0] for x in sample) / sum(x[1] for x in sample))
    estimates.sort()
    return [estimates[int(.025 * repetitions)], estimates[int(.975 * repetitions)]]


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('D:/r2ai'))
    ap.add_argument('--inspect', action='store_true')
    args = ap.parse_args()
    t0 = time.monotonic()
    root, sub = args.root, args.root / 'data/sub'
    paths = {**{k: sub / v for k, v in SPECS.items()},
             'doclist': sub / 'vi37/doclist_v37.json',
             'exchanges': sub / 'vi38/exchanges_vi38b.json'}
    lock = root / 'locks/heavy_job.lock'
    if args.inspect:
        print(json.dumps({'lock': lock.read_text() if lock.exists() else 'absent',
                          'files': {k: p.stat().st_size if p.exists() else None for k, p in paths.items()}}, indent=2))
        return
    if lock.exists():
        raise SystemExit('STOP: heavy_job.lock exists; no survey writes')
    docs = {q: list(map(int, ds)) for q, ds in load(paths['doclist']).items()}
    exchanges = load(paths['exchanges'])
    assert len(docs) == 1200
    assert sum(map(len, exchanges.values())) == 2692
    groups = {'vi37': [(q, d) for q, ds in docs.items() for d in ds],
              'A_out': [(q, int(x['out'])) for q, xs in exchanges.items() for x in xs],
              'B_in': [(q, int(x['incoming'])) for q, xs in exchanges.items() for x in xs]}
    for q, xs in exchanges.items():
        assert all(int(x['out']) in docs[q] and int(x['incoming']) not in docs[q] for x in xs)
    fh = load(paths['fhyde'])
    fr = {q: ranked(rows) for q, rows in fh.items()}
    for lo, hi in [(1, 30), (31, 99), (100, 169), (170, 250), (251, 400), (401, 10000)]:
        groups[f'vi37_fh_{lo}_{hi}'] = [(q, d) for q, d in groups['vi37'] if lo <= fr[q].get(d, 10001) <= hi]
    targets = set(groups['vi37'] + groups['A_out'] + groups['B_in'])
    records = {key: {'qid': key[0], 'doc_id': key[1]} for key in targets}
    out = sub / 'label_origin_survey'
    out.mkdir(parents=True, exist_ok=True)
    report = {'groups': {g: {'pairs': len(xs)} for g, xs in groups.items()},
              'sources': {}, 'paired': {}, 'inputs': {}, 'public_subset_known': False}
    report['swap_precision_gap_proxy'] = {
        'mean_m_over_n_all1200': sum(len(xs) / len(docs[q]) for q, xs in exchanges.items()) / 1200,
        'delta_macro_P_rounded': .0001,
        'warning': 'DeltaP divided by this weight assumes public subset is representative; not individual A/B precision.'}
    for name, path in paths.items():
        if not path.exists():
            report['inputs'][name] = {'missing': True}
            continue
        raw = path.read_bytes()
        report['inputs'][name] = {'path': str(path), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        if name not in SPECS:
            continue
        cache = fh if name == 'fhyde' else json.loads(raw)
        ranks = {q: ranked(rows) for q, rows in cache.items()}
        cap = max(map(len, ranks.values()))
        for (q, d), row in records.items():
            row[name] = ranks.get(q, {}).get(d)
        report['sources'][name] = {'max_depth': cap, 'queries': len(ranks),
                                  'min_depth': min(map(len, ranks.values())),
                                  'top30_in_vi37_mean': sum(sum(d in docs[q] for d, r in rs.items() if r <= 30)
                                                           for q, rs in ranks.items()) / len(docs)}
        report['sources'][name]['top30_outside_vi37_mean'] = sum(
            sum(d not in docs[q] for d, rank in rs.items() if rank <= 30)
            for q, rs in ranks.items()) / len(docs)
        for group, pairs in groups.items():
            report['groups'][group][name] = summarize([ranks.get(q, {}).get(d) for q, d in pairs], cap)
        paired = {}
        for k in KS:
            deltas = []
            for q, xs in exchanges.items():
                if xs:
                    rs = ranks.get(q, {})
                    delta = sum(int(rs.get(int(x['incoming']), cap + 1) <= k) -
                                int(rs.get(int(x['out']), cap + 1) <= k) for x in xs)
                    deltas.append((delta, len(xs)))
            paired[str(k)] = {'B_minus_A_fraction': sum(x[0] for x in deltas) / 2692,
                              'query_bootstrap95': paired_bootstrap(deltas)}
        report['paired'][name] = paired
        del cache, ranks, raw
        if time.monotonic() - t0 > 105:
            raise SystemExit('STOP: 105-second CPU budget exceeded; partial per-doc data not written')
    # A compact per-document result makes every top-k membership auditable.
    out_ids, in_ids = set(groups['A_out']), set(groups['B_in'])
    fields = ['qid', 'doc_id', 'in_vi37', 'swap_role', *SPECS]
    with gzip.open(out / 'per_doc_ranks.csv.gz', 'wt', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key, row in sorted(records.items(), key=lambda kv: (int(kv[0][0]), kv[0][1])):
            writer.writerow(dict(row, in_vi37=int(key[1] in docs[key[0]]),
                                 swap_role='A' if key in out_ids else 'B' if key in in_ids else ''))
    report['seconds'] = round(time.monotonic() - t0, 3)
    report['source_combinations'] = {}
    for group in ('vi37', 'A_out', 'B_in'):
        pairs = groups[group]
        counts = {'dense200_only_vs_bm25200': 0, 'bm25200_only_vs_dense200': 0,
                  'both200': 0, 'neither200': 0}
        for key in pairs:
            row = records[key]
            b, d = (row.get(s) is not None and row[s] <= 200 for s in ('bm25', 'dense'))
            field = 'both200' if b and d else 'bm25200_only_vs_dense200' if b else 'dense200_only_vs_bm25200' if d else 'neither200'
            counts[field] += 1
        report['source_combinations'][group] = counts
    report['note'] = 'All 1200 queries, not public-only. No gold labels. Absent cache ranks are censored.'
    (out / 'survey.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'seconds': report['seconds'],
                      'pairs': {k: len(v) for k, v in groups.items()}, 'sources': list(report['sources'])}))


if __name__ == '__main__':
    main()
