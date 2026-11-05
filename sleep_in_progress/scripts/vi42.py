"""CPU/cache-only vi37 plus midpoint bridges. Never loads models or submits."""
import argparse
from bisect import bisect_left
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

import dwmax_diag as dw
from chunking import chunk_document
from reorder_zip import write_reordered_zip
import vi23
import vi29
import vi37
from vi39 import compare_by_id, wait_lock

ROOT = vi37.ROOT
OUT = ROOT / 'data/sub/vi42'
LIMIT = 99_000_000
WORD = re.compile(r'\S+')


def midpoint(tokens, start, end):
    """Nearest token-start to character midpoint; lower offset wins ties."""
    choices = [s for s, e in tokens if start <= s < end]
    assert choices
    return min(choices, key=lambda s: (abs(2*s-start-end), s))


def bridge_geometry(text):
    chunks = chunk_document(text, 'vi', max_units=900)
    tokens = [m.span() for m in WORD.finditer(text)]
    starts = [s for s, e in tokens]
    out = []
    for i, (left, right) in enumerate(zip(chunks, chunks[1:])):
        p = right.start  # whitespace gap belongs to neither chunk
        j = bisect_left(starts, p)
        lo, hi = max(0, j-300), min(len(tokens), j+300)
        s = midpoint(tokens, left.start, left.end)
        e = midpoint(tokens, right.start, right.end)
        ct = text[s:e].strip()
        assert ct and ct in text
        out.append({'index': i, 'start': s, 'end': e, 'text': ct,
                    'band': (tokens[lo][0], tokens[hi-1][1]), 'boundary': p})
    return chunks, out


def gate_score(windows, band):
    lo, hi = band
    scores = [float(lg) for s, e, lg in windows if min(e, hi) > max(s, lo)]
    assert all(math.isfinite(x) for x in scores)
    score = max(scores, default=None)
    return score if score is not None and score >= 0 else None


def select_bridges(item, docs, windows, geometry=None):
    """Keep original chunks untouched; top-50 doc order is the vi37 fhyde doclist."""
    selected = {}
    seen = set()
    for c in item['relevant_chunks']:
        d = str(c['doc_id'])
        selected.setdefault(d, set()).add(c['chunk_text'])
        seen.add((d, c['chunk_text']))
    out = []
    geometry = geometry if geometry is not None else {}
    for rank, did in enumerate(item['relevant_docs'][:50], 1):
        d = str(did)
        if d not in selected:
            continue
        if d not in geometry:
            geometry[d] = bridge_geometry(docs[int(d)])
        chunks, bridges = geometry[d]
        assert selected[d] <= {c.text for c in chunks}, ('selected U900 grid mismatch', item['id'], d)
        if not bridges:
            continue
        assert d in windows, ('missing cached windows', item['id'], d)
        for b in bridges:
            i = b['index']
            if chunks[i].text not in selected[d] and chunks[i+1].text not in selected[d]:
                continue
            sc = gate_score(windows[d], b['band'])
            if sc is None or (d, b['text']) in seen:
                continue
            seen.add((d, b['text']))
            out.append({'doc_id': did, 'chunk_text': b['text'], 'score': sc,
                        'rank': rank, 'boundary_index': i, 'start': b['start'], 'end': b['end']})
    return sorted(out, key=lambda b: (-b['score'], b['rank'], b['boundary_index']))


def append_bridges(ref, bridges, threshold=None, enabled=True):
    return [{**it, 'relevant_chunks': list(it['relevant_chunks']) + [
        {'doc_id': b['doc_id'], 'chunk_text': b['chunk_text']}
        for b in bridges.get(str(it['id']), [])
        if enabled and (threshold is None or b['score'] >= threshold)]} for it in ref]


def budget_threshold(scores, measure, limit=LIMIT):
    """Binary search whole score groups, preserving ties; measure actual reordered ZIP."""
    levels = sorted(set(scores))
    tried = {}
    def get(i):
        if i not in tried:
            tried[i] = measure(levels[i] if i < len(levels) else None)
            ordered = sorted(tried)
            for a, b in zip(ordered, ordered[1:]):
                if tried[a] <= limit < tried[b]:
                    raise RuntimeError('nonmonotone zip-fit at score groups')
        return tried[i]
    # None here means no bridges (only if all groups must be removed).
    if not levels:
        assert get(0) <= limit
        return None, tried
    if get(0) <= limit:
        return levels[0], tried
    lo, hi = 0, len(levels)
    assert get(hi) <= limit, 'baseline alone exceeds budget'
    while hi-lo > 1:
        mid = (lo+hi)//2
        if get(mid) <= limit:
            hi = mid
        else:
            lo = mid
    assert get(hi) <= limit < get(hi-1)
    return levels[hi] if hi < len(levels) else None, tried


def window_grid(text):
    spans = [m.span() for m in WORD.finditer(text)]
    out = []
    for i in range(0, len(spans), 250):
        end = min(i+300, len(spans))
        out.append((spans[i][0], spans[end-1][1]))
        if end == len(spans):
            break
    return out


def load_windows(needed):
    owners = dw.read(vi37.DIAG / 'docwin_max_provenance.json')
    result = {q: {} for q in needed}
    fresh = dw.read(vi37.OUT / 'docwin_new.json.zst')
    for q, ds in needed.items():
        result[q].update({d: fresh[q][d] for d in ds if d in fresh.get(q, {})})
    del fresh
    used = Counter()
    for name, rel, legacy in dw.layer_specs():
        keys = [(q, d) for q, ds in needed.items() for d in ds
                if d not in result[q] and owners.get(q, {}).get(d) == name]
        if not keys:
            continue
        obj = dw.read(ROOT / 'data/sub' / rel)
        for q, d in keys:
            result[q][d] = obj[q][d]
            used[name] += 1
        del obj
    missing = [(q, d) for q, ds in needed.items() for d in ds if d not in result[q]]
    assert not missing, missing[:10]
    return result, dict(used)


def offline(ref, bridges, threshold=None):
    nq = len(ref)
    rows = [b for bs in bridges.values() for b in bs if threshold is None or b['score'] >= threshold]
    ranks = Counter(str(b['rank']) for b in rows)
    return {'bridges': len(rows), 'bridges_per_query': len(rows)/nq,
            'chunks_per_query': (sum(len(it['relevant_chunks']) for it in ref)+len(rows))/nq,
            'length_syllables': dw.dist([len(b['chunk_text'].split()) for b in rows]),
            'docs_with_bridges_per_query': sum(len({str(b['doc_id']) for b in bs
                if threshold is None or b['score'] >= threshold}) for bs in bridges.values())/nq,
            'doc_rank_counts': dict(sorted(ranks.items(), key=lambda kv: int(kv[0]))),
            'doc_rank_buckets': {f'{lo}-{lo+9}': sum(n for r, n in ranks.items() if lo <= int(r) < lo+10)
                                 for lo in range(1, 51, 10)}}


def fingerprint(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return {'bytes': Path(path).stat().st_size, 'sha256': h.hexdigest()}


def build():
    start = time.monotonic()
    inputs = [vi37.OUT/'submission_vi37.json', vi37.OUT/'docwin_new.json.zst',
              vi37.OUT/'cache_fhyde_v37.json', vi37.DIAG/'docwin_max_provenance.json']
    before = {str(p): fingerprint(p) for p in inputs}
    ref = dw.read(inputs[0])
    sanity = compare_by_id(append_bridges(ref, {}, enabled=False), ref)
    assert sanity == {'count': 1200, 'docs': 1200, 'chunks': 1200}
    dw.write(OUT/'sanity_report.json', sanity)
    needed = {}
    for it in ref:
        have = {str(c['doc_id']) for c in it['relevant_chunks']}
        needed[str(it['id'])] = set(map(str, it['relevant_docs'][:50])) & have
    # Reuse the exact full final-text export already validated by vi37, never crawl.
    from rerank_run import load_docs
    wanted = {int(c['doc_id']) for it in ref for c in it['relevant_chunks']}
    files = sorted((vi37.OUT/'verify_text').glob('docs_verify.part*.jsonl.zst'))
    assert files, 'vi37 full text export missing'
    docs = load_docs([str(p) for p in files], wanted=wanted)
    assert wanted <= set(docs), 'missing exported documents'
    wins, used = load_windows(needed)
    adjusted = []
    for q, ds in wins.items():
        for d, ws in ds.items():
            grid = window_grid(docs[int(d)])
            if [(s, e) for s, e, lg in ws] != grid:
                # Historical Unicode offset drift: only ordinal remap if the exact
                # 300/250 window COUNT agrees; scores and their order remain fixed.
                assert len(ws) == len(grid), ('window count mismatch', q, d, len(ws), len(grid))
                ds[d] = [[s, e, w[2]] for (s, e), w in zip(grid, ws)]
                adjusted.append([q, d])
    geometry = {}
    bridges = {str(it['id']): select_bridges(it, docs, wins[str(it['id'])], geometry) for it in ref}
    vi37.writez(OUT/'bridges_vi42.json.zst', bridges)
    pre = offline(ref, bridges)
    dw.write(OUT/'selection_report.json', {'sanity': sanity, 'before_budget': pre,
             'historical_layers': used, 'ordinal_offset_remap_pairs': adjusted})
    print('SELECTED', json.dumps(pre), flush=True)
    del wins, geometry
    attempts = []
    zp = OUT/'submission_vi42.zip'
    def measure(thr):
        sub = append_bridges(ref, bridges, thr, enabled=thr is not None)
        info = write_reordered_zip(sub, zp)
        attempts.append({'threshold': thr, **info})
        dw.write(OUT/'budget_attempts.json', attempts)
        print('BUDGET', json.dumps(attempts[-1]), flush=True)
        return info['zip_bytes']
    threshold, _ = budget_threshold([b['score'] for bs in bridges.values() for b in bs], measure)
    sub = append_bridges(ref, bridges, threshold, enabled=threshold is not None)
    info = write_reordered_zip(sub, zp)
    assert info['zip_bytes'] <= LIMIT
    ordered = OUT/'submission_vi42.json'
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ['submission.json'] and z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        assert z.testzip() is None
        with z.open('submission.json') as src, ordered.open('wb') as dst:
            shutil.copyfileobj(src, dst)
    subprocess.run([sys.executable, str(Path(__file__).parent/'evaluate.py'), 'validate', str(ordered)], check=True)
    code = vi23.check_zip(str(ordered), str(zp), str(OUT/'report_chain_vi42.json'), 'vi42', max_zip=LIMIT)
    assert code == 0
    assert compare_by_id(dw.read(ordered), sub) == {'count': 1200, 'docs': 1200, 'chunks': 1200}
    cmp = compare_by_id(sub, ref)
    assert cmp['count'] == cmp['docs'] == 1200
    assert all(it['relevant_chunks'][:len(old['relevant_chunks'])] == old['relevant_chunks'] for it, old in zip(sub, ref))
    bad = set(map(int, dw.read(ROOT/'data/sub/diag28/bad_ids.json'))) - set(map(int, dw.read(ROOT/'data/sub/diag30/new_ids.json')))
    corpus = vi29.load_corpus_ids(ROOT/'data/sub/diag19/links_corpus_vi_en_zh.parquet')
    clean = vi29.assert_clean(sub, bad, corpus)
    verbatim = sum(c['chunk_text'] not in docs[int(c['doc_id'])] for it in sub for c in it['relevant_chunks'])
    assert verbatim == 0
    assert before == {str(p): fingerprint(p) for p in inputs}, 'source changed during build'
    post = offline(ref, bridges, threshold) if threshold is not None else offline(ref, {})
    report = {**info, **fingerprint(zp), 'sanity': sanity, 'doc_compare': cmp,
              'vi37_chunks_prefix_preserved': 1200, 'assert_clean': clean, 'verbatim_bad': verbatim,
              'validate': 0, 'check_zip': code, 'crc': 0, 'threshold': threshold,
              'attempts': attempts, 'before_budget': pre, 'after_budget': post,
              'ordinal_offset_remap_pairs': adjusted, 'source_fingerprints': before,
              'seconds': time.monotonic()-start,
              'budget_note': 'whole-score groups; binary search assumes monotone ZIP fit, adjacent boundary measured if cut'}
    dw.write(OUT/'build_report_vi42.json', report)
    print('DONE', json.dumps(report), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['build', 'status'], default='build', nargs='?')
    args = ap.parse_args()
    if args.stage == 'status':
        lock = ROOT/'locks/heavy_job.lock'
        print('LOCK', lock.read_text() if lock.exists() else 'absent')
        for p in sorted(OUT.glob('*report.json')):
            print(p.name, p.stat().st_size)
        p = OUT/'vi42.log'
        if p.exists():
            with p.open('rb') as f:
                f.seek(max(0, p.stat().st_size-1200))
                print(f.read().decode('utf-8', 'replace')[-1200:])
        return
    OUT.mkdir(parents=True, exist_ok=True)
    with wait_lock(ROOT/'locks/heavy_job.lock', f'vi42 pid={os.getpid()}'):
        build()


if __name__ == '__main__':
    main()
