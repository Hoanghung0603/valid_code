"""CPU-only read-only survey of vi37; writes only a separate llmj_survey report.

Token estimates are ceil(chars/3), NOT measured Qwen tokens. No GPU/API calls.
The head+window figures are conservative budget caps, not observed full-window lengths.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def describe(values):
    xs = sorted(values)
    if not xs:
        return {'n': 0}
    return {'n': len(xs), 'sum': sum(xs), 'mean': sum(xs) / len(xs),
            **{f'p{p}': xs[round((len(xs)-1)*p/100)] for p in (10, 50, 90, 95, 99)},
            'max': xs[-1]}


def regions(hits, doclist, chunkdocs, nq):
    ranks = {int(h[0]): i+1 for i, h in enumerate(hits)}
    dl = set(map(int, doclist))
    bands = {'pool': list(ranks), 'doclist': list(dl)}
    for lo, hi in [(1, 30), (31, 99), (100, 169), (170, 250), (251, 400)]:
        bands[f'rank_{lo}_{hi}'] = [d for d, r in ranks.items() if lo <= r <= hi]
    bands['rank_401_plus'] = [d for d, r in ranks.items() if r > 400]
    bands['a_100_nq'] = [d for d in dl if 100 <= ranks.get(d, 0) <= nq]
    bands['a_100_nq_nochunk'] = [d for d in bands['a_100_nq'] if d not in chunkdocs]
    bands['below_nq_30'] = [d for d, r in ranks.items() if nq < r <= nq+30 and d not in dl]
    bands['below_nq_50'] = [d for d, r in ranks.items() if nq < r <= nq+50 and d not in dl]
    active = bool(bands['a_100_nq_nochunk'])
    bands['c_swap30'] = bands['a_100_nq_nochunk'] + (bands['below_nq_30'] if active else [])
    bands['c_swap50'] = bands['a_100_nq_nochunk'] + (bands['below_nq_50'] if active else [])
    bands['rrf_top_nq30_incomplete'] = list(dict.fromkeys(
        [d for d, r in ranks.items() if r <= nq+30] + list(dl)))
    return bands


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path('D:/r2ai'))
    ap.add_argument('--out', type=Path)
    ap.add_argument('--status', action='store_true')
    ap.add_argument('--fill-lengths', action='store_true', help='read missing text from guarded vi36 index')
    a = ap.parse_args()
    if a.status:
        lock = a.root / 'locks/heavy_job.lock'
        print('LOCK', lock.read_text() if lock.exists() else 'absent')
        for name in ['survey.json', 'supplemental_lengths.json']:
            p = a.root / 'data/sub/llmj_survey' / name
            print(name, (p.stat().st_size, p.stat().st_mtime) if p.exists() else 'absent')
        return
    t0 = time.monotonic()
    subroot = a.root / 'data/sub'
    v = subroot / 'vi37'
    if (a.root / 'locks/heavy_job.lock').exists():
        raise SystemExit('STOP: another heavy job owns the lock; survey did not acquire it')
    fh = read(v / 'cache_fhyde_v37.json')
    dl = read(v / 'doclist_v37.json')
    pool = read(subroot / 'vi36/in1/pool_v36.json')
    lengths = read(subroot / 'diag37/doc_lengths.json')
    lengths.update(read(v / 'export_lengths.json'))
    supplemental = subroot / 'llmj_survey/supplemental_lengths.json'
    if supplemental.exists():
        lengths.update(read(supplemental))
    if a.fill_lengths:
        import dwmax_diag as dw
        from bigindex import BigIndex
        from boilerplate import BoilerplateFilter
        from export_docs import iter_all_texts
        index = a.root / 'data/index/vi36'
        dw.safe_index(index)
        wanted = {int(h[0]) for rows in pool.values() for h in rows if str(h[0]) not in lengths}
        extra = {}
        with dw.owned_lock(a.root / 'locks/heavy_job.lock', 'llmj_survey missing lengths'):
            idx = BigIndex(index)
            positions = {int(d): i for i, d in enumerate(idx.ids) if int(d) in wanted}
            bp = BoilerplateFilter.from_json((index / 'boilerplate.json').read_text(encoding='utf-8'))
            for d, _, text in iter_all_texts(idx, positions, bp, 200000, no_title=True):
                extra[str(d)] = {'chars': len(text), 'windows': dw.window_count(text)}
                if time.monotonic()-t0 > 95:
                    print('STOP LENGTH SCAN at 95s; report partial coverage', flush=True)
                    break
        supplemental.parent.mkdir(parents=True, exist_ok=True)
        saved = read(supplemental) if supplemental.exists() else {}
        saved.update(extra)
        supplemental.write_text(json.dumps(saved), encoding='utf-8')
        lengths.update(extra)
    q8b = read(v / 'q8b_v37.json')
    qt = {str(r['id']): r['query'] for r in map(json.loads,
          (v / 'modal_input/queries.jsonl').read_text(encoding='utf-8').splitlines())}
    # Read the finished submission only to identify protected chunk-source docs.
    submission = read(v / 'submission_vi37.json')
    chunkdocs = {str(r['id']): {int(c['doc_id']) for c in r['relevant_chunks']} for r in submission}
    del submission
    stats = {}
    nqs = []
    for q, hits in fh.items():
        nq = min(400, max(40, sum(h[2] > -2 for h in pool[q])))
        nqs.append(nq)
        have8b = {int(h[0]) for h in q8b[q]}
        for name, ds in regions(hits, dl[q], chunkdocs[q], nq).items():
            st = stats.setdefault(name, {'pairs': 0, 'unique': set(), 'perquery': [],
                 'chars': [], 'tokens_est': [], 'missing_lengths': 0, 'empty': 0,
                 'missing_8b': 0, 'head1024_prompt': 0, 'head2048_prompt': 0,
                 'head256_win512_prompt_cap': 0})
            st['pairs'] += len(ds)
            st['perquery'].append(len(ds))
            st['unique'].update(ds)
            for d in ds:
                st['missing_8b'] += d not in have8b
                row = lengths.get(str(d))
                if row is None:
                    st['missing_lengths'] += 1
                    continue
                chars = row['chars']
                tok = math.ceil(chars/3)
                st['empty'] += chars == 0
                st['chars'].append(chars)
                st['tokens_est'].append(tok)
                overhead = 256 + math.ceil(len(qt[q])/3)
                for field, cap in [('head1024_prompt', 1024), ('head2048_prompt', 2048),
                                   ('head256_win512_prompt_cap', 768)]:
                    st[field] += min(tok, cap) + overhead
    for st in stats.values():
        st['unique'] = len(st['unique'])
        st['perquery'] = describe(st['perquery'])
        st['chars'] = describe(st['chars'])
        rawtokens = st.pop('tokens_est')
        st['tokens_est'] = describe(rawtokens)
        st['head1024_doc_tokens_est'] = describe([min(t, 1024) for t in rawtokens])
        st['head2048_doc_tokens_est'] = describe([min(t, 2048) for t in rawtokens])
        st['head256_win512_doc_token_cap'] = describe([min(t, 768) for t in rawtokens])
    # Actual maximum-window offsets for the new vi37 layer only; never impute old scores.
    import zstandard
    with (v / 'docwin_new.json.zst').open('rb') as f:
        with zstandard.ZstdDecompressor().stream_reader(f) as z:
            wins = json.load(z)
    fresh = []
    for q, hits in fh.items():
        nq = min(400, max(40, sum(h[2] > -2 for h in pool[q])))
        ds = regions(hits, dl[q], chunkdocs[q], nq)['c_swap30']
        for d in ds:
            ws = wins.get(q, {}).get(str(d), [])
            if not ws:
                continue
            s, e, _ = max(ws, key=lambda w: w[2])
            chars = lengths[str(d)]['chars']
            headend = min(chars, 768)  # 256 estimated tokens
            end = min(e, s+1536)  # 512 estimated tokens, chosen max window
            unionchars = headend + end-s - max(0, min(headend, end)-s)
            fresh.append(math.ceil(unionchars/3) + 256 + math.ceil(len(qt[q])/3))
    report = {'version': 1, 'token_estimator': 'ceil(chars/3), sensitivity +/-30%; overhead 256+query chars/3',
              'nq': describe(nqs), 'regions': stats,
              'fresh_layer_swap30_actual_offsets_prompt_tokens_est': describe(fresh),
              'seconds': round(time.monotonic()-t0, 2),
              'inputs_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                  [v/'doclist_v37.json', v/'cache_fhyde_v37.json']}}
    out = a.out or subroot / 'llmj_survey/survey.json'
    assert 'llmj_survey' in str(out), 'write only to new survey directory'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'out': str(out), 'seconds': report['seconds'],
          'regions': {k: {'pairs': s['pairs'], 'missing_lengths': s['missing_lengths'],
                          'tokens_cap': s['head256_win512_prompt_cap']} for k, s in stats.items()}}), flush=True)


if __name__ == '__main__':
    main()
