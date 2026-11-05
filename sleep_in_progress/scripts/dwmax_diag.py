"""CPU-only diag37: cached document-window MaxP, coverage and fixed RRF simulation.

No model loading, network, submission or Modal invocation. Only main owns the lock.
--prepare-export is a separate, NOT automatically executed CPU stage for step 2.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path
import re
import statistics
import time


def read(path):
    path = Path(path)
    if path.suffix == '.zst':
        import zstandard
        with path.open('rb') as f:
            with zstandard.ZstdDecompressor().stream_reader(f) as z:
                return json.loads(z.read())
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


@contextmanager
def owned_lock(path, label):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as f:
        f.write(label)
    try:
        yield
    finally:
        if path.exists() and path.read_text(encoding='utf-8') == label:
            path.unlink()


def mean(xs):
    return statistics.mean(xs) if xs else None


def dist(xs):
    xs = sorted(xs)
    def pct(p):
        x = (len(xs) - 1) * p
        i = int(x)
        return xs[i] + (xs[min(i + 1, len(xs) - 1)] - xs[i]) * (x - i)
    return {'n': len(xs), 'mean': mean(xs), **(
        {'p10': pct(.1), 'p50': pct(.5), 'p90': pct(.9), 'min': xs[0], 'max': xs[-1]}
        if xs else {})}


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[i]] == values[order[j]]:
            j += 1
        for k in range(i, j):
            result[order[k]] = (i + j - 1) / 2
        i = j
    return result


def pearson(a, b):
    if len(a) < 2:
        return None
    ma, mb = mean(a), mean(b)
    den = (sum((v - ma)**2 for v in a) * sum((v - mb)**2 for v in b)) ** .5
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / den if den else None


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def window_count(text):
    """Exactly docwin_run.window_spans 300/250, without materializing offsets."""
    n = sum(1 for _ in re.finditer(r'\S+', text[:200000]))
    return 0 if not n else 1 + max(0, math.ceil((n - 300) / 250))


def maxima(wins):
    return {str(q): {str(d): max(float(w[2]) for w in rows if w[2] is not None)
                     for d, rows in ds.items() if rows and any(w[2] is not None for w in rows)}
            for q, ds in wins.items()}


def merge_layer(merged, owners, incoming, name, excluded, allowed):
    """Latest VALID layer wins; never maximize across versions of a text."""
    stats = Counter()
    overlaps = {}
    for q, ds in incoming.items():
        for d, score in ds.items():
            if d in excluded or d not in allowed.get(q, set()):
                stats['excluded'] += 1
                continue
            stats['accepted'] += 1
            old = merged.setdefault(q, {})
            if d in old:
                key = owners[q][d] + ' -> ' + name
                overlaps.setdefault(key, []).append((old[d], score))
                stats['overlap'] += 1
            old[d] = score
            owners.setdefault(q, {})[d] = name
    return dict(stats), overlaps


def overlap_report(pairs):
    a, b = [x for x, _ in pairs], [y for _, y in pairs]
    delta = [abs(x-y) for x, y in pairs]
    return {'pairs': len(pairs), 'pearson': pearson(a, b), 'max_abs': max(delta, default=None),
            'mean_abs': mean(delta), 'different': sum(x != y for x, y in pairs)}


def coverage(docs, scores):
    total = sum(map(len, docs.values()))
    have = sum(d in scores.get(q, {}) for q, ds in docs.items() for d in ds)
    return {'pairs': total, 'scored': have, 'missing': total-have,
            'pct': 100 * have / total if total else None}


def correlation_report(doclists, scores, source):
    vals, top, pair_count = [], [], 0
    for q, docs in doclists.items():
        sm = {str(h[0]): float(h[2]) for h in source.get(q, []) if h[2] is not None}
        ds = [d for d in docs if d in scores.get(q, {}) and d in sm]
        pair_count += len(ds)
        v = spearman([scores[q][d] for d in ds], [sm[d] for d in ds])
        if v is not None:
            vals.append(v)
        if ds:
            k = min(30, len(ds))
            ta = set(sorted(ds, key=lambda d: -scores[q][d])[:k])
            tb = set(sorted(ds, key=lambda d: -sm[d])[:k])
            top.append(len(ta & tb) / k)
    return {'queries': len(vals), 'common_pairs': pair_count, 'spearman_macro': mean(vals),
            'spearman_distribution': dist(vals), 'top30_overlap_fraction_macro': mean(top),
            'note': 'Intersection-only; top-k min(30, common docs), no imputed missing ranks'}


def simulate(pool, q8b, dense, hyde, scores, nq, ref_docs):
    from fuse3 import fuse3
    dw = {}
    for q, ds in scores.items():
        # Deterministic ties by original pool order, as fuse3 stable sorting.
        order = {str(h[0]): i for i, h in enumerate(pool[q])}
        dw[q] = [[int(d), None, s] for d, s in
                 sorted(ds.items(), key=lambda kv: (-kv[1], order[kv[0]]))]
    fused = fuse3(pool, q8b, dense, (1., 2., 2., 1.), 60, hyde, [(dw, 1.)])
    out = {}
    for q, hits in fused.items():
        docs = [str(h[0]) for h in hits[:nq[q]]]
        docs.extend(d for d in ref_docs.get(q, []) if d not in set(docs))
        out[q] = docs
    return out, fused


def comparison(old, new, fhyde, controls):
    adds, drops, changes, ordered_changes, jaccards = [], [], 0, 0, []
    rank_in, rank_out = [], []
    agreements = {name: {'in': [0, 0], 'out': [0, 0]} for name in controls}
    per_query = {}
    for q, ds in old.items():
        a, b = set(ds), set(new[q])
        added, removed = b-a, a-b
        adds.append(len(added))
        drops.append(len(removed))
        changes += bool(added or removed)
        ordered_changes += ds != new[q]
        jaccards.append(len(a & b) / len(a | b) if a | b else 1.)
        rr = {str(h[0]): i for i, h in enumerate(fhyde[q], 1)}
        rank_in.extend(rr[d] for d in added if d in rr)
        rank_out.extend(rr[d] for d in removed if d in rr)
        per_query[q] = {'in': sorted(added, key=lambda d: rr.get(d, 10**9)),
                        'out': sorted(removed, key=lambda d: rr.get(d, 10**9))}
        for name, ctrl in controls.items():
            c = set(ctrl.get(q, []))
            for label, docs in [('in', added), ('out', removed)]:
                agreements[name][label][0] += len(docs & c)
                agreements[name][label][1] += len(docs)
    for groups in agreements.values():
        for label, (n, total) in groups.items():
            groups[label] = {'matches': n, 'total': total, 'fraction': n/total if total else None}
    return {'queries_set_changed': changes, 'queries_order_changed': ordered_changes,
            'added_per_query': dist(adds), 'removed_per_query': dist(drops),
            'symmetric_difference_mean': mean([a+b for a, b in zip(adds, drops)]),
            'jaccard_macro': mean(jaccards), 'incoming_original_rank': dist(rank_in),
            'outgoing_original_rank': dist(rank_out), 'unlabeled_controls': agreements}, per_query


def layer_specs():
    return [('top100', 'docwin/docwin_top100.json.zst', True),
            ('ext', 'docwin_ext/docwin_ext.json.zst', True),
            ('p3', 'diag18/docwin_p3/docwin_p3.json.zst', True),
            ('vi23', 'diag23/docwin_vi23.json.zst', True),
            ('vi24', 'diag24/docwin_vi24.json.zst', True),
            ('vi25', 'diag25/docwin_vi25.json.zst', True),
            ('vi30', 'diag30/docwin_vi30.json.zst', False),
            ('vi31', 'vi31/docwin_vi31.json.zst', False),
            ('vi32', 'vi32/docwin_vi31.json.zst', False),
            ('vi33', 'vi33/docwin_vi31.json.zst', False),
            ('vi34', 'vi34/docwin_vi31.json.zst', False),
            ('vi36', 'vi36/docwin_vi36.json.zst', False)]


def safe_index(index):
    """Hard guard: do not read any shard under data/crawl, even via index metadata."""
    meta = read(index / 'meta.json')
    for path in meta['shards']:
        if '/data/crawl/' in str(path).replace('\\', '/').lower():
            raise RuntimeError('Forbidden crawl shard: ' + path)


def text_lengths(root, wanted):
    from bigindex import BigIndex
    from boilerplate import BoilerplateFilter
    from export_docs import iter_all_texts
    index = root / 'data/index/vi36'
    safe_index(index)
    idx = BigIndex(index)
    pos = {int(d): i for i, d in enumerate(idx.ids)}
    bp = BoilerplateFilter.from_json((index / 'boilerplate.json').read_text(encoding='utf-8'))
    lengths = {}
    for d, _i, text in iter_all_texts(idx, {d: pos[d] for d in wanted if d in pos}, bp,
                                     200000, no_title=True, log=lambda m: print(m, flush=True)):
        lengths[str(d)] = {'chars': len(text), 'windows': window_count(text)}
    return lengths


def prepare_export(root, out):
    """Only explicit step 2 CPU command; no Modal launch or nested lock."""
    import pyarrow.parquet as pq
    import zstandard
    from bigindex import BigIndex
    from boilerplate import BoilerplateFilter
    from export_docs import iter_all_texts
    cands = read(out / 'cands_docwin_missing_top250.json')
    index = root / 'data/index/vi36'
    safe_index(index)
    idx = BigIndex(index)
    wanted = {int(h[0]): int(h[1]) for rows in cands.values() for h in rows}
    bp = BoilerplateFilter.from_json((index / 'boilerplate.json').read_text(encoding='utf-8'))
    dest = out / 'modal_input'
    dest.mkdir(parents=True, exist_ok=True)
    n, part, size, f, z = 0, -1, 0, None, None
    try:
        for d, i, text in iter_all_texts(idx, wanted, bp, 200000, no_title=True):
            raw = (json.dumps({'id': d, 'idx': i, 'text': text}, ensure_ascii=False)+'\n').encode('utf-8')
            if z is None or size >= 70_000_000:
                if z is not None:
                    z.close()
                    f.close()
                part += 1
                f = (dest / f'docs_docwin.part{part:03d}.jsonl.zst').open('wb')
                z = zstandard.ZstdCompressor(level=6).stream_writer(f, closefd=False)
                size = 0
            z.write(raw)
            size += len(raw)
            n += 1
    finally:
        if z is not None:
            z.close()
            f.close()
    if n != len(wanted):
        raise RuntimeError(f'Missing export texts: {len(wanted)-n}')
    write(dest / 'cands_docwin.json', cands)
    qs = pq.read_table(root / 'data/query.parquet').to_pydict()
    with (dest / 'queries.jsonl').open('w', encoding='utf-8') as f:
        for q, text in zip(qs['id'], qs['query']):
            f.write(json.dumps({'id': int(q), 'query': text}, ensure_ascii=False)+'\n')
    write(dest / 'export_report.json', {'docs': n, 'parts': part+1, 'modal_launched': False})


def run(root, out):
    t0 = time.time()
    subroot = root / 'data/sub'
    source = subroot / 'vi36/in1'
    pool, q8b, dense, hyde = [read(source / name) for name in
                            ['pool_v36.json', 'q8b_v36.json', 'dense_v36.json', 'hyde_v36.json']]
    from fuse3 import fuse3
    from union_sub import adaptive_n
    fhyde = fuse3(pool, q8b, dense, (1., 2., 2., 1.), 60, hyde)
    cached = read(subroot / 'vi36/cache_fhyde_v36.json')
    allowed = {q: {str(h[0]) for h in hits} for q, hits in pool.items()}
    cache_mismatches = sum([h for h in cached[q] if str(h[0]) in allowed[q]] != hits
                           for q, hits in fhyde.items())
    # Stage doclist fuses raw sources, while stage_build fuses in1 after filtering.
    # Filtering before RRF can shift source ranks; submission exact match below is authoritative.
    print('RAW_VS_CLEAN_FHYDE', cache_mismatches, flush=True)
    nq = {q: adaptive_n(hits, -2., 40, 400) for q, hits in pool.items()}
    refs = read(source / 'ref_chunks_v36.json')
    ref_docs = {str(it['id']): list(dict.fromkeys(str(c['doc_id']) for c in it['relevant_chunks']))
                for it in refs}
    submission = read(subroot / 'vi36/submission_vi36.json')
    doclists = {str(it['id']): [str(d) for d in it['relevant_docs']] for it in submission}
    chunk_docs = {str(it['id']): {str(c['doc_id']) for c in it['relevant_chunks']} for it in submission}
    del submission
    baseline = {}
    for q, hits in fhyde.items():
        ds = [str(h[0]) for h in hits[:nq[q]]]
        ds.extend(d for d in ref_docs.get(q, []) if d not in set(ds))
        baseline[q] = ds
    assert baseline == doclists, 'Baseline doclist must reproduce vi36 submission exactly'
    bad = set(map(str, read(subroot / 'diag28/bad_ids.json')))
    scores, owners, layers, overlaps = {}, {}, [], {}
    for name, rel, legacy in layer_specs():
        path = subroot / rel
        incoming = maxima(read(path))
        st, ov = merge_layer(scores, owners, incoming, name, bad if legacy else set(), allowed)
        for k, pairs in ov.items():
            overlaps.setdefault(k, []).extend(pairs)
        layers.append({'name': name, 'path': str(path), 'legacy_bad_excluded': legacy, **st})
        print('LAYER', name, st, flush=True)
    write(out / 'docwin_max_current.json', scores)
    write(out / 'docwin_max_provenance.json', owners)
    write(out / 'cache_fhyde_vi36_clean.json', fhyde)
    cov = {'doclist_vi36': coverage(doclists, scores)}
    bands = {k: {} for k in ['1-100', '101-n_q', 'n_q+1-250', '251-400', 'top250']}
    missing = {}
    for q, hits in fhyde.items():
        for key in bands:
            bands[key][q] = []
        for r, h in enumerate(hits, 1):
            d = str(h[0])
            predicates = {'1-100': r <= 100, '101-n_q': 101 <= r <= nq[q],
                          'n_q+1-250': nq[q] < r <= 250, '251-400': 251 <= r <= 400,
                          'top250': r <= 250}
            for key, yes in predicates.items():
                if yes:
                    bands[key][q].append(d)
            if r <= 250 and d not in scores.get(q, {}):
                missing.setdefault(q, []).append([h[0], h[1]])
    cov.update({k: coverage(ds, scores) for k, ds in bands.items()})
    write(out / 'cands_docwin_missing_top250.json', missing)
    corr = {name: correlation_report(doclists, scores, cache) for name, cache in
            [('fhyde', fhyde), ('bge', pool), ('q8b', q8b), ('dense', dense), ('hyde', hyde)]}
    groups = {k: [] for k in ['head_1_30', 'tail_gt100', 'tail_with_chunk', 'tail_without_chunk']}
    tail_total = tail_chunk = tail_missing = tail_missing_chunk = 0
    for q, ds in doclists.items():
        rr = {str(h[0]): r for r, h in enumerate(fhyde[q], 1)}
        for d in ds:
            r = rr.get(d, 10**9)
            score = scores.get(q, {}).get(d)
            if r > 100:
                tail_total += 1
                tail_chunk += d in chunk_docs[q]
                tail_missing += score is None
                tail_missing_chunk += score is None and d in chunk_docs[q]
            if score is None:
                continue
            if r <= 30:
                groups['head_1_30'].append(score)
            if r > 100:
                groups['tail_gt100'].append(score)
                groups['tail_with_chunk' if d in chunk_docs[q] else 'tail_without_chunk'].append(score)
    tail = {k: dist(v) for k, v in groups.items()}
    tail.update({'total_tail': tail_total, 'tail_chunk_docs': tail_chunk,
                 'tail_chunk_fraction': tail_chunk/tail_total, 'missing_tail': tail_missing,
                 'missing_tail_with_chunk': tail_missing_chunk})
    # Descriptive bins fixed in advance, not a threshold search.
    tail['fixed_logit_bins'] = {}
    for lo, hi in [(-math.inf, -2), (-2, 0), (0, 2), (2, math.inf)]:
        a = sum(lo <= v < hi for v in groups['tail_with_chunk'])
        b = sum(lo <= v < hi for v in groups['tail_without_chunk'])
        tail['fixed_logit_bins'][f'[{lo},{hi})'] = {'pairs': a+b, 'with_chunk': a,
                                                  'chunk_fraction': a/(a+b) if a+b else None}
    controls = {'base_7415': read(subroot / 'runner/full1200b/doclist_full1200b.json'),
                'p1_7414': read(subroot / 'runner/full1200/doclist_full1200.json'),
                'vi34': read(subroot / 'vi34/doclist_vi34.json')}
    controls = {k: {q: list(map(str, ds)) for q, ds in v.items()} for k, v in controls.items()}
    proposed, _ = simulate(pool, q8b, dense, hyde, scores, nq, ref_docs)
    simulation, exchanges = comparison(doclists, proposed, fhyde, controls)
    simulation['missing_pool_pairs'] = sum(len(ds)-len(scores.get(q, {})) for q, ds in allowed.items())
    simulation['missing_proposed_doclist'] = coverage(proposed, scores)
    simulation['new_docs_missing_q8b'] = sum(d not in {str(h[0]) for h in q8b[q]}
                                             for q, x in exchanges.items() for d in x['in'])
    write(out / 'doclist_vi37_partial_offline.json', proposed)
    write(out / 'doclist_exchanges.json', exchanges)
    wanted = {int(d) for ds in doclists.values() for d in ds}
    wanted.update(int(h[0]) for rows in missing.values() for h in rows)
    print('TEXT_LENGTHS', len(wanted), flush=True)
    lengths = text_lengths(root, wanted)
    write(out / 'doc_lengths.json', lengths)
    missing_pairs = [str(h[0]) for rows in missing.values() for h in rows]
    n_windows = sum(lengths[d]['windows'] for d in missing_pairs if d in lengths)
    missing_stats = {'pairs': len(missing_pairs), 'unique_docs': len(set(missing_pairs)),
                     'estimated_windows': n_windows,
                     'missing_text_docs': len(set(missing_pairs)-set(lengths)),
                     'empty_text_pairs': sum(lengths.get(d, {}).get('windows') == 0 for d in missing_pairs)}
    absent_sub = [(q, d) for q, ds in doclists.items() for d in ds if d not in scores.get(q, {})]
    missing_stats['doclist_missing_empty_text_pairs'] = sum(
        lengths.get(d, {}).get('windows') == 0 for q, d in absent_sub)
    missing_stats['doclist_missing_nonempty_text_pairs'] = sum(
        lengths.get(d, {}).get('windows', 0) > 0 for q, d in absent_sub)
    missing_stats['doclist_missing_above_rank250'] = sum(
        r > 250 and str(h[0]) in set(doclists[q]) and str(h[0]) not in scores.get(q, {})
        for q, hits in fhyde.items() for r, h in enumerate(hits, 1))
    bias = []
    for q, docs in doclists.items():
        ds = [d for d in docs if d in lengths and d in scores.get(q, {})]
        c = spearman([lengths[d]['windows'] for d in ds], [scores[q][d] for d in ds])
        if c is not None:
            bias.append(c)
    tail['length_windows_vs_max_spearman_macro'] = mean(bias)
    seconds = n_windows * 2377 / 1116226
    # Runner scores are separate: useful existing cache, but never mixed into primary simulation.
    runner_scores = maxima(read(subroot / 'runner/full1200/docwin.json.zst'))
    runner_overlap = [(s, runner_scores[q][d]) for q, ds in scores.items() for d, s in ds.items()
                      if d in runner_scores.get(q, {})]
    runner_reuse = sum(d in runner_scores.get(q, {}) for q, rows in missing.items() for d, _ in
                       [(str(h[0]), h[1]) for h in rows])
    rep = {'definition': {'model': 'BAAI/bge-reranker-v2-m3', 'window_words': 300, 'stride': 250,
                          'max_len_tokens_query_plus_window': 512, 'text_char_cap': 200000,
                          'rule': 'Latest valid layer wins; pre-vi30 discard ALL diag28 bad_ids; pool clean filter; no cross-layer max',
                          'runner_not_merged': True, 'vi35_not_merged': True},
           'baseline': {'queries': len(pool), 'doclist_exact_reproduction': True,
                        'filtered_cached_fhyde_mismatches': cache_mismatches, 'nq': dist(list(nq.values()))},
           'layers': layers, 'layer_overlaps': {k: overlap_report(v) for k, v in overlaps.items()},
           'layer_overlap_aggregate': overlap_report([p for pairs in overlaps.values() for p in pairs]),
           'coverage': cov, 'missing_top250': missing_stats, 'correlations': corr, 'tail': tail,
           'simulation_partial_only': simulation,
           'runner_supplement': {'overlap': overlap_report(runner_overlap),
                                 'missing_top250_already_in_runner': runner_reuse,
                                 'note': 'Not reused without text/version compatibility checks'},
           'step2': {'gpu_seconds': seconds, 'gpu_usd_at_1_95_hour': seconds*1.95/3600,
                     'wall_minutes_one_gpu': seconds/60, 'safety_30pct_gpu_seconds': seconds*1.3,
                     'safety_30pct_gpu_usd': seconds*1.3*1.95/3600,
                     'workspace': __import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"), 'launched': False,
                     'rate_unit': 'query-window pairs, not query-document pairs',
                     'benchmark': '1,116,226 windows / 2377 s / 1 L40S; excludes CPU/RAM/startup charges'},
           'elapsed_seconds': time.time()-t0}
    write(out / 'dwmax_diag.json', rep)
    print('DONE', str(out / 'dwmax_diag.json'), round(time.time()-t0, 1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('D:/r2ai'))
    ap.add_argument('--prepare-export', action='store_true')
    args = ap.parse_args()
    out = args.root / 'data/sub/diag37'
    with owned_lock(args.root / 'locks/heavy_job.lock', f'dwmax_diag pid={os.getpid()}'):
        if args.prepare_export:
            prepare_export(args.root, out)
        else:
            run(args.root, out)


if __name__ == '__main__':
    main()
