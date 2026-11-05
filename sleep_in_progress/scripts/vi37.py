"""vi37: vi36 + full-pool docwin MaxP RRF x1 for DOC only. Never submit.

CPU stages own one lock; Modal is invoked separately. Existing vi*/q8b* are import-only.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dwmax_diag as dw
import vi36
import vi29 as v29
import q8b31

ROOT = Path(os.environ.get('R2AI_ROOT', 'D:/r2ai'))
OUT = ROOT / 'data/sub/vi37'
DIAG = ROOT / 'data/sub/diag37'
BASE = ROOT / 'data/sub/vi36/in1'


def inputs():
    return {key: dw.read(BASE / name) for key, name in
            [('pool', 'pool_v36.json'), ('q8b', 'q8b_v36.json'), ('dense', 'dense_v36.json'),
             ('hyde', 'hyde_v36.json')]}


def writez(path, obj):
    import zstandard
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as f:
        with zstandard.ZstdCompressor(level=6).stream_writer(f) as z:
            z.write(json.dumps(obj, ensure_ascii=False).encode('utf-8'))


def docwin_hits(pool, scores):
    """Ranks within pool, stable pool-order tie break; empty docs have no contribution."""
    out = {}
    for q, hits in pool.items():
        rows = [[h[0], h[1], scores.get(q, {}).get(str(h[0]))] for h in hits]
        out[q] = sorted([h for h in rows if h[2] is not None], key=lambda h: -h[2])
    return out


@contextmanager
def chain_docwin(scores, enabled=True):
    """Intercept ONLY fhyde call made inside unchanged vi29.chain_build_v29."""
    import fuse3
    import vi23
    original, old_w, old_nq = fuse3.fuse3, vi23.W_Q8B_FHYDE, vi23.NQ_FUSED
    def fuse(pool, q8b, dense, weights=(1., 1., 1.), k=60, bm25=None, extras=None):
        ex = list(extras or [])
        if enabled and len(weights) > 3 and weights[3]:
            ex.append((docwin_hits(pool, scores), 1.))
        return original(pool, q8b, dense, weights, k, bm25, ex)
    fuse3.fuse3 = fuse
    vi23.W_Q8B_FHYDE, vi23.NQ_FUSED = 2., None
    try:
        yield
    finally:
        fuse3.fuse3 = original
        vi23.W_Q8B_FHYDE, vi23.NQ_FUSED = old_w, old_nq


def fused_doclist(data, scores):
    from fuse3 import fuse3
    from union_sub import adaptive_n
    import vi23
    fh = fuse3(data['pool'], data['q8b'], data['dense'], (1., 2., 2., 1.), 60,
               data['hyde'], [(docwin_hits(data['pool'], scores), 1.)])
    ref = {str(it['id']): it['relevant_chunks'] for it in dw.read(BASE / 'ref_chunks_v36.json')}
    dl = {q: vi23.full8b_docs(vi23.first_n_docs(hits, adaptive_n(data['pool'][q], -2., 40, 400)),
                             ref.get(q, [])) for q, hits in fh.items()}
    return fh, dl


class Parts:
    """Bound uncompressed bytes, so SFTP parts cannot exceed 100MiB."""
    def __init__(self, dest, prefix):
        self.dest, self.prefix = Path(dest), prefix
        self.dest.mkdir(parents=True, exist_ok=True)
        self.files, self.size, self.z, self.f = [], 0, None, None
    def add(self, row):
        import zstandard
        raw = (json.dumps(row, ensure_ascii=False)+'\n').encode('utf-8')
        if self.z is None or self.size + len(raw) > 70_000_000:
            self.close()
            path = self.dest / f'{self.prefix}.part{len(self.files):03d}.jsonl.zst'
            self.files.append(path)
            self.f = path.open('wb')
            self.z = zstandard.ZstdCompressor(level=6).stream_writer(self.f, closefd=False)
            self.size = 0
        self.z.write(raw)
        self.size += len(raw)
    def close(self):
        if self.z is not None:
            self.z.close()
            self.f.close()
            self.z, self.f = None, None


def query_texts():
    import pyarrow.parquet as pq
    qs = pq.read_table(ROOT / 'data/query.parquet').to_pydict()
    return {str(q): text for q, text in zip(qs['id'], qs['query'])}


def export_texts(wanted, dest, prefix, cap=200000, keep=None):
    from bigindex import BigIndex
    from boilerplate import BoilerplateFilter
    from export_docs import iter_all_texts
    index = ROOT / 'data/index/vi36'
    dw.safe_index(index)
    idx = BigIndex(index)
    pos = {int(d): i for i, d in enumerate(idx.ids)}
    assert not set(wanted)-set(pos), 'Missing index IDs'
    bp = BoilerplateFilter.from_json((index / 'boilerplate.json').read_text(encoding='utf-8'))
    writer = Parts(dest, prefix)
    lengths, texts = {}, {}
    try:
        for d, i, text in iter_all_texts(idx, {d: pos[d] for d in wanted}, bp, 200000, no_title=True):
            text = text[:cap]
            writer.add({'id': d, 'idx': i, 'text': text})
            lengths[str(d)] = {'chars': len(text), 'windows': dw.window_count(text), 'idx': i}
            if keep is not None and d in keep:
                texts[d] = text
    finally:
        writer.close()
    assert len(lengths) == len(wanted), 'Missing exported text'
    return lengths, texts, writer.files


def stage_export():
    """Full pool export; fresh nonempty pairs only, plus 300 cached window bench pairs."""
    data = inputs()
    scores, owners = dw.read(DIAG / 'docwin_max_current.json'), dw.read(DIAG / 'docwin_max_provenance.json')
    pool = data['pool']
    missing = {q: [h[:2] for h in hits if str(h[0]) not in scores.get(q, {})] for q, hits in pool.items()}
    known = [(q, d, owners[q][d]) for q, ds in scores.items() for d in ds]
    bench_keys = random.Random(3737).sample(sorted(known), 300)
    needed_layers = {layer for _, _, layer in bench_keys}
    windows = {}
    for name, rel, _ in dw.layer_specs():
        if name not in needed_layers:
            continue
        wins = dw.read(ROOT / 'data/sub' / rel)
        for q, d, owner in bench_keys:
            if owner == name:
                windows[(q, d)] = max(wins[q][d], key=lambda w: w[2])
    wanted = {int(h[0]) for rows in missing.values() for h in rows}
    wanted.update(int(d) for q, d, _ in bench_keys)
    dest = OUT / 'modal_input'
    lengths, bench_text, files = export_texts(wanted, dest, 'docs_docwin', keep={int(d) for q, d, _ in bench_keys})
    empty = {d for d, st in lengths.items() if not st['windows']}
    fresh = {q: [h for h in rows if str(h[0]) not in empty] for q, rows in missing.items()}
    shards = [{q: [] for q in pool} for _ in range(3)]
    # Split WITHIN each query: modal docwin_run top-docs500 cannot silently truncate a large pool.
    for q, rows in fresh.items():
        for i, h in enumerate(rows):
            shards[i % 3][q].append(h)
    qt = query_texts()
    with (dest / 'queries.jsonl').open('w', encoding='utf-8') as f:
        for q, text in qt.items():
            f.write(json.dumps({'id': int(q), 'query': text}, ensure_ascii=False)+'\n')
    bench = []
    for q, d, _ in bench_keys:
        s, e, score = windows[(q, d)]
        text = bench_text[int(d)]
        assert e <= len(text) and text[s:e].strip(), (q, d, 'window/text mismatch')
        bench.append({'qid': q, 'doc_id': int(d), 'query': qt[q], 'text': text[s:e], 'score': score})
    dw.write(dest / 'bench_docwin_in.json', bench)
    for i, shard in enumerate(shards):
        dw.write(dest / f'cands_docwin_shard{i}.json', {q: rows for q, rows in shard.items() if rows})
    dw.write(OUT / 'missing_fullpool.json', missing)
    dw.write(OUT / 'fresh_docwin.json', fresh)
    dw.write(OUT / 'empty_doc_ids.json', sorted(empty))
    dw.write(OUT / 'export_lengths.json', lengths)
    n_windows = sum(lengths[str(h[0])]['windows'] for rows in fresh.values() for h in rows)
    sec = n_windows*2377/1116226
    rep = {'missing_pairs': sum(map(len, missing.values())), 'fresh_pairs': sum(map(len, fresh.values())),
           'empty_pairs': sum(str(h[0]) in empty for rows in missing.values() for h in rows),
           'unique_exported_docs': len(lengths), 'windows': n_windows, 'gpu_seconds_est': sec,
           'gpu_usd_est': sec*1.95/3600, 'bench': 300,
           'shard_pairs': [sum(map(len, s.values())) for s in shards],
           'max_query_pairs_per_shard': max(len(rows) for s in shards for rows in s.values()),
           'files': [p.name for p in sorted(dest.iterdir()) if p.is_file()]}
    assert rep['max_query_pairs_per_shard'] <= 500
    assert rep['gpu_usd_est']*1.3 < 4., 'Reserve >=$1 for 8B/startup; stop before launch'
    dw.write(OUT / 'export_report.json', rep)
    print(json.dumps(rep), flush=True)


def stage_sanity():
    files = v29.chain_files(BASE, vi36.FILE_NAMES)
    with chain_docwin({}, enabled=False):
        rep = v29.run_chain('vi37_sanity', 56, files, 'doc_text', OUT)
    import vi23
    cmp = vi23.compare_subs(dw.read(OUT / 'submission_vi37_sanity.json'),
                            str(ROOT / 'data/sub/vi36/submission_vi36.json'))
    v29.require_sanity(cmp)
    dw.write(OUT / 'sanity_report.json', cmp)
    (OUT / 'submission_vi37_sanity.json').unlink()
    print('SANITY', json.dumps(cmp), flush=True)


def stage_dwmerge():
    bench = dw.read(OUT / 'modal_input/bench_docwin_in.json')
    got = dw.read(OUT / 'modal_out/bench_docwin_out.json')['scores']
    assert len(got) == len(bench) == 300
    report = dw.overlap_report([(r['score'], s) for r, s in zip(bench, got)])
    assert report['pearson'] >= .999, report
    scores = dw.read(DIAG / 'docwin_max_current.json')
    fresh = dw.read(OUT / 'fresh_docwin.json')
    newwins = {}
    for i in range(3):
        ob = dw.read(OUT / f'modal_out/docwin_shard{i}.json.zst')
        for q, ds in ob.items():
            overlap = set(ds) & set(newwins.get(q, {}))
            assert not overlap
            newwins.setdefault(q, {}).update(ds)
    absent = [(q, h[0]) for q, rows in fresh.items() for h in rows if str(h[0]) not in newwins.get(q, {})]
    assert not absent, absent[:5]
    for q, ds in dw.maxima(newwins).items():
        scores.setdefault(q, {}).update(ds)
    writez(OUT / 'docwin_new.json.zst', newwins)
    dw.write(DIAG / 'docwin_max_v37.json', scores)
    data = inputs()
    empty = set(dw.read(OUT / 'empty_doc_ids.json'))
    total = sum(map(len, data['pool'].values()))
    empty_pairs = sum(str(h[0]) in empty for rows in data['pool'].values() for h in rows)
    scored = sum(str(h[0]) in scores.get(q, {}) for q, rows in data['pool'].items() for h in rows)
    pct = scored/(total-empty_pairs)
    assert pct >= .995, pct
    dw.write(OUT / 'dwmerge_report.json', {'bench': report, 'pool_pairs': total,
             'scored': scored, 'empty_pairs': empty_pairs, 'coverage_nonempty': pct,
             'unscored_nonempty': total-empty_pairs-scored,
             'new_pairs': sum(map(len, newwins.values())),
             'new_windows': sum(len(ws) for ds in newwins.values() for ws in ds.values())})
    fh, dl = fused_doclist(data, scores)
    dw.write(OUT / 'cache_fhyde_pre.json', fh)
    dw.write(OUT / 'doclist_pre.json', dl)


def stage_qselect():
    scores = dw.read(DIAG / 'docwin_max_v37.json')
    data = inputs()
    if (OUT / 'q8b_v37.json').exists():
        data['q8b'] = dw.read(OUT / 'q8b_v37.json')
    fh, dl = fused_doclist(data, scores)
    targets = vi36.select_8b_targets(dl, fh, data['q8b'], 30)
    n = sum(map(len, targets.values()))
    merged_round = (dw.read(OUT / 'q8b_merge_report.json').get('merged_round', 1)
                    if (OUT / 'q8b_v37.json').exists() else 0)
    suffix = f'round{merged_round + 1}'
    dw.write(OUT / f'q8b_targets_{suffix}.json', targets)
    dw.write(OUT / f'q8b_select_{suffix}.json', {'pairs': n, 'queries': len(targets)})
    if merged_round:
        dw.write(OUT / 'cache_fhyde_v37.json', fh)
        dw.write(OUT / 'doclist_v37.json', dl)
        print('Q8B_' + suffix.upper(), n, flush=True)
        return
    bench = q8b31.bench_sample(fh, data['q8b'], 200, seed=37, top=400)
    wanted = {d for ds in targets.values() for d in ds} | {d for q, d, s in bench}
    dest = OUT / 'q8b_input'
    lengths, _, files = export_texts(wanted, dest, 'docs_q8b', cap=q8b31.CAP)
    pos = {int(d): st['idx'] for d, st in lengths.items()}
    shards = q8b31.split_shards(targets, 2)
    for i, sh in enumerate(shards):
        dw.write(dest / f'cands_q8b_shard{i}.json', {q: [[d, pos[d]] for d in ds] for q, ds in sh.items()})
    bc = {}
    for q, d, s in bench:
        bc.setdefault(q, []).append([d, pos[d]])
    dw.write(dest / 'cands_q8b_bench.json', bc)
    dw.write(dest / 'bench_q8b_old.json', bench)
    (dest / 'queries.jsonl').write_bytes((OUT / 'modal_input/queries.jsonl').read_bytes())
    estimate = (n+len(bench))/9.*1.95/3600
    dw.write(OUT / 'q8b_export_report.json', {'pairs': n, 'bench': len(bench),
             'shard_pairs': [sum(map(len, s.values())) for s in shards],
             'gpu_usd_est': estimate, 'files': [p.name for p in sorted(dest.iterdir()) if p.is_file()]})
    print('Q8B_ROUND1', n, 'gpu_usd_est', estimate, flush=True)


def stage_qexport(round_number=2):
    assert round_number in (2, 3)
    targets = dw.read(OUT / f'q8b_targets_round{round_number}.json')
    n = sum(map(len, targets.values()))
    assert 0 < n <= (14 if round_number == 2 else 4), n
    dest = OUT / 'q8b_input'
    wanted = {int(d) for ds in targets.values() for d in ds}
    lengths, _, files = export_texts(wanted, dest, f'docs_q8b_r{round_number}', cap=q8b31.CAP)
    assert all(st['chars'] for st in lengths.values()), 'Empty supplementary 8B text'
    cands = {q: [[int(d), lengths[str(d)]['idx']] for d in ds] for q, ds in targets.items()}
    cp = dest / f'cands_q8b_r{round_number}.json'
    dw.write(cp, cands)
    dw.write(OUT / f'q8b_export_round{round_number}.json', {
        'pairs': n, 'queries': len(targets), 'files': [cp.name] + [p.name for p in files],
        'fingerprints': {p.name: {'bytes': p.stat().st_size,
            'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in [cp, *files]}})
    print('QEXPORT', round_number, n, flush=True)


def checked_q8b(obj, expected):
    assert set(obj) == set(expected), 'q8b query mismatch'
    for q, rows in obj.items():
        assert len(rows) == len(expected[q])
        assert {str(h[0]) for h in rows} == {str(h[0]) for h in expected[q]}
        assert all(h[2] is not None and math.isfinite(h[2]) for h in rows)


def stage_qmerge(round_number=1):
    assert round_number in (1, 2, 3)
    data = inputs()
    new = {}
    for i in range(2):
        obj = dw.read(OUT / f'modal_out/q8b_shard{i}_out.json')
        expected = dw.read(OUT / f'q8b_input/cands_q8b_shard{i}.json')
        checked_q8b(obj, expected)
        for q, rows in obj.items():
            new.setdefault(q, []).extend(rows)
    round_pairs = {'1': sum(map(len, new.values()))}
    for r in range(2, round_number + 1):
        obj = dw.read(OUT / f'modal_out/q8b_r{r}_out.json')
        expected = dw.read(OUT / f'q8b_input/cands_q8b_r{r}.json')
        targets = dw.read(OUT / f'q8b_targets_round{r}.json')
        assert {q: [str(h[0]) for h in rows] for q, rows in expected.items()} == {
            q: [str(d) for d in ds] for q, ds in targets.items()}
        checked_q8b(obj, expected)
        round_pairs[str(r)] = sum(map(len, obj.values()))
        for q, rows in obj.items():
            assert not {str(h[0]) for h in rows} & {str(h[0]) for h in new.get(q, [])}
            new.setdefault(q, []).extend(rows)
    bench = dw.read(OUT / 'q8b_input/bench_q8b_old.json')
    got = {q: {str(h[0]): h[2] for h in rows} for q, rows in
           dw.read(OUT / 'modal_out/q8b_bench_out.json').items()}
    pairs = [(s, got[q][str(d)]) for q, d, s in bench if str(d) in got.get(q, {})]
    rep = dw.overlap_report(pairs)
    assert len(pairs) == 200 and rep['pearson'] >= .999, rep
    out, st = q8b31.merge_q8b(data['q8b'], new)
    check = q8b31.check_merged(out, new, data['pool'])
    assert check['ok'], check
    dw.write(OUT / 'q8b_v37.json', out)
    dw.write(OUT / 'q8b_merge_report.json', {'bench': rep, 'merge': st, 'check': check,
              'new_pairs': sum(map(len, new.values())), 'merged_round': round_number,
              'round_pairs': round_pairs})


def require_q8b_complete():
    r = dw.read(OUT / 'q8b_merge_report.json').get('merged_round', 1) + 1
    assert dw.read(OUT / f'q8b_select_round{r}.json')['pairs'] == 0, f'Stop: round{r} needs approval'


def stage_chunk():
    from rerank_run import load_docs
    from chunks_from_docwin import build_q_rows, select_q_rows, verify_verbatim
    from vi24 import merge_chunks
    require_q8b_complete()
    old = dw.read(BASE / 'cands_v36.json.zst')
    dl = dw.read(OUT / 'doclist_v37.json')
    pool = inputs()['pool']
    pos = {q: {str(h[0]): h[1] for h in hits} for q, hits in pool.items()}
    # Build only newly required documents lacking existing candidate chunks. Scores from the
    # new full-pool layer or valid historical layers; original candidates remain unchanged.
    fresh = {}
    for q, ds in dl.items():
        have = {str(r[0]) for r in old.get(q, [])}
        fresh[q] = [[int(d), pos[q][str(d)]] for d in ds if str(d) not in have]
    wanted = {h[0] for rows in fresh.values() for h in rows}
    # Texts may include historical scores whose chunks were previously capped away.
    lengths, _, files = export_texts(wanted, OUT / 'chunk_text', 'docs_chunk')
    docs = load_docs([str(p) for p in files], wanted=wanted)
    wins = dw.read(OUT / 'docwin_new.json.zst')
    for name, rel, legacy in dw.layer_specs():
        obj = dw.read(ROOT / 'data/sub' / rel)
        # Only take the exact provenance owner used by diag37, avoiding stale text.
        owners = dw.read(DIAG / 'docwin_max_provenance.json') if name == 'top100' else owners
        for q, rows in fresh.items():
            for d, _ in rows:
                sd = str(d)
                if sd not in wins.get(q, {}) and owners.get(q, {}).get(sd) == name:
                    if sd in obj.get(q, {}):
                        wins.setdefault(q, {})[sd] = obj[q][sd]
    absent = [(q, d) for q, rows in fresh.items() for d, i in rows
              if docs[d].strip() and str(d) not in wins.get(q, {})]
    assert not absent, absent[:5]
    cache = {}
    new = {q: select_q_rows(build_q_rows(rows, docs, wins.get(q, {}), 900, 1000, cache), 100000)
           for q, rows in fresh.items()}
    assert verify_verbatim(new, docs) == 0
    merged = merge_chunks(old, new)
    writez(OUT / 'cands_v37.json.zst', merged)
    dw.write(OUT / 'chunk_report.json', {'new_pairs': sum(map(len, fresh.values())),
             'empty_pairs': sum(not docs[d].strip() for rows in fresh.values() for d, i in rows),
             'new_chunks': sum(map(len, new.values())), 'verbatim_bad': 0,
             'total_candidates': sum(map(len, merged.values()))})


def stage_build():
    require_q8b_complete()
    expected = dw.read(OUT / 'doclist_v37.json')
    files = v29.chain_files(BASE, vi36.FILE_NAMES)
    files['q8b'], files['cands'] = OUT / 'q8b_v37.json', OUT / 'cands_v37.json.zst'
    scores = dw.read(DIAG / 'docwin_max_v37.json')
    oldout = v29.OUT
    v29.OUT = OUT
    try:
        with chain_docwin(scores):
            delta, attempts = v29.find_delta(files, 'doc_text', 99000000, 56, 'vi37p')
            check = v29.chain_zip_check('vi37', delta, 99000000, files, 'doc_text', validate=True)
    finally:
        v29.OUT = oldout
    assert check['code'] == 0, check
    sub = dw.read(OUT / 'submission_vi37.json')
    excl = set(map(int, dw.read(ROOT / 'data/sub/diag28/bad_ids.json'))) - set(
        map(int, dw.read(ROOT / 'data/sub/diag30/new_ids.json')))
    corpus = v29.load_corpus_ids(ROOT / 'data/sub/diag19/links_corpus_vi_en_zh.parquet')
    clean = v29.assert_clean(sub, excl, corpus)
    assert {str(it['id']): it['relevant_docs'] for it in sub} == expected
    from rerank_run import load_docs
    wanted = {int(c['doc_id']) for it in sub for c in it['relevant_chunks']}
    _, _, files_text = export_texts(wanted, OUT / 'verify_text', 'docs_verify')
    docs = load_docs([str(p) for p in files_text], wanted=wanted)
    verbatim_bad = sum(c['chunk_text'] not in docs[int(c['doc_id'])]
                       for it in sub for c in it['relevant_chunks'])
    assert verbatim_bad == 0, verbatim_bad
    zp = OUT / 'submission_vi37.zip'
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ['submission.json']
        assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
    sha = hashlib.sha256(zp.read_bytes()).hexdigest()
    dw.write(OUT / 'build_report_vi37.json', {'delta': delta, 'attempts': attempts,
             'check_zip': check, 'validate': 'actual evaluate.py validate exited 0',
              'assert_clean': clean, 'verbatim_bad': verbatim_bad,
              'zip_bytes': zp.stat().st_size, 'sha256': sha})


def count_capped(cands, rrf, doclists, kq, delta):
    """Count strict C_q < k_q+delta after doc filtering and (doc,text) dedup."""
    return sum(len(v29.order_q_docdedup(cands.get(q, []), rrf.get(q, []),
                                       set(doclists[q]), 'doc_text')) < kq[q] + delta
               for q in doclists)


def stage_stats():
    sub = dw.read(OUT / 'submission_vi37.json')
    ref = dw.read(ROOT / 'data/sub/vi36/submission_vi36.json')
    sm, rm = {str(it['id']): it for it in sub}, {str(it['id']): it for it in ref}
    old = {q: list(map(str, it['relevant_docs'])) for q, it in rm.items()}
    new = {q: list(map(str, it['relevant_docs'])) for q, it in sm.items()}
    comp, exchanges = dw.comparison(old, new, dw.read(DIAG / 'cache_fhyde_vi36_clean.json'), {})
    rows = {}
    for name, items in [('vi36', ref), ('vi37', sub)]:
        rep = dw.read(ROOT / f'data/sub/{name}/report_chain_{name}.json')
        rows[name] = {'docs_per_query': sum(len(it['relevant_docs']) for it in items)/1200,
                      'chunks_per_query': sum(len(it['relevant_chunks']) for it in items)/1200,
                      'chunk_source_docs_per_query': sum(len({str(c['doc_id']) for c in
                          it['relevant_chunks']}) for it in items)/1200,
                      'delta': rep['delta'], 'nq_mean': rep['n_q_TB']}
    kq = {str(it['id']): len(it['relevant_chunks']) for it in dw.read(v29.REF_K)}
    data = inputs()
    for name, doclists, candpath, q8b in [
            ('vi36', old, BASE / 'cands_v36.json.zst', data['q8b']),
            ('vi37', new, OUT / 'cands_v37.json.zst', dw.read(OUT / 'q8b_v37.json'))]:
        rrf = __import__('fuse3').fuse3(data['pool'], q8b, data['dense'], (1., 1., 2., 0.), 60)
        rows[name]['queries_capped_by_Cq'] = count_capped(
            dw.read(candpath), rrf, doclists, kq, rows[name]['delta'])
        rows[name]['ref_source_docs_added_per_query'] = rows[name]['docs_per_query'] - rows[name]['nq_mean']
    top30 = sum(len(set(old[q][:30]) & set(new[q][:30])) for q in sm)/1200
    chunk_overlap = sum(len({(str(c['doc_id']), c['chunk_text']) for c in sm[q]['relevant_chunks']} &
                            {(str(c['doc_id']), c['chunk_text']) for c in rm[q]['relevant_chunks']})
                        for q in sm)/1200
    dw.write(OUT / 'stats_report_vi37.json', {'rows': rows, 'doclist_comparison': comp,
             'queries_doclist_set_unchanged': 1200 - comp['queries_set_changed'],
             'queries_doclist_order_unchanged': 1200 - comp['queries_order_changed'],
             'top30_overlap_mean': top30, 'chunk_exact_overlap_mean': chunk_overlap})
    dw.write(OUT / 'doclist_exchanges.json', exchanges)


def status():
    lock = ROOT / 'locks/heavy_job.lock'
    print('LOCK', lock.read_text() if lock.exists() else 'absent')
    for p in sorted(OUT.glob('*report*.json')):
        print(p.name, p.stat().st_size)
    for p in sorted(OUT.glob('*.log')):
        with p.open('rb') as f:
            f.seek(max(0, p.stat().st_size-2000))
            print(p.name, '\n'.join(f.read().decode('utf-8', 'replace').splitlines()[-5:]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['export', 'sanity', 'dwmerge', 'qselect', 'qexport', 'qmerge', 'chunk',
                                       'build', 'stats', 'status'])
    ap.add_argument('--prepare-export', action='store_true')
    ap.add_argument('--round', type=int, choices=[1, 2, 3], default=None)
    a = ap.parse_args()
    if a.stage == 'status':
        status()
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with dw.owned_lock(ROOT / 'locks/heavy_job.lock', f'vi37 {a.stage} pid={os.getpid()}'):
        if a.stage in ('qmerge', 'qexport'):
            globals()['stage_'+a.stage](a.round or (1 if a.stage == 'qmerge' else 2))
        else:
            globals()['stage_'+a.stage]()
    print('DONE', a.stage, flush=True)


if __name__ == '__main__':
    main()
