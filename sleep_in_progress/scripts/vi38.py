"""vi38 CPU stages. Only relevant_docs bytes may change; vi34--37 are read-only."""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dwmax_diag as dw
from llm_judge_run import MODEL, REVISION, VLLM, chat, parse_logprobs, sha, snippets

ROOT = Path(os.environ.get('R2AI_ROOT', 'D:/r2ai'))
OUT = ROOT/'data/sub/vi38'
OLD = ROOT/'data/sub/vi37'


def fingerprint(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return {'bytes': Path(p).stat().st_size, 'sha256': h.hexdigest()}


def select_pairs(fh, pool, submission):
    selected, controls, nqs = [], {'top30': [], 'tail': []}, {}
    for item in submission:
        q = str(item['id'])
        docs = set(map(int, item['relevant_docs']))
        protected = {int(c['doc_id']) for c in item['relevant_chunks']}
        allowed = {int(h[0]) for h in pool[q]}
        assert docs <= allowed and len(docs) == len(item['relevant_docs'])
        nq = min(400, max(40, sum(h[2] > -2 for h in pool[q])))
        nqs[q] = nq
        aa, bb = [], []
        for rank, h in enumerate(fh[q], 1):
            d = int(h[0])
            row = dict(qid=q, doc_id=d, rank=rank, fhyde=h[2], pair_id=f'{q}:{d}')
            if 100 <= rank <= nq and d in docs and d not in protected:
                aa.append(dict(row, role='A'))
            if nq < rank <= nq+30 and d not in docs:
                bb.append(dict(row, role='B'))
            if rank <= 30:
                controls['top30'].append(dict(row, role='top30'))
            if rank > max(nq+30, len(fh[q])-60):
                controls['tail'].append(dict(row, role='tail'))
        selected.extend(aa)
        if aa:
            selected.extend(bb)
    return selected, controls, nqs


def stage_select(a):
    fh = dw.read(OLD/'cache_fhyde_v37.json')
    pool = dw.read(ROOT/'data/sub/vi36/in1/pool_v36.json')
    sub = dw.read(OLD/'submission_vi37.json')
    rows, controls, nqs = select_pairs(fh, pool, sub)
    dw.write(OUT/'selected.json', rows)
    dw.write(OUT/'controls.json', controls)
    dw.write(OUT/'nq.json', nqs)
    counts = Counter(r['role'] for r in rows)
    dw.write(OUT/'select_report.json', dict(counts, total=len(rows), plan=99877,
             delta_from_plan=len(rows)-99877, queries=len({r['qid'] for r in rows}),
             submission=fingerprint(OLD/'submission_vi37.json'),
             fhyde=fingerprint(OLD/'cache_fhyde_v37.json')))
    print('SELECT', dict(counts), len(rows), flush=True)


def stratified_bench(rows, controls, lengths, queries):
    """256 per region; cycle query-length x document-length quartiles, no duplicates."""
    groups = {'A': [r for r in rows if r['role'] == 'A'],
              'B': [r for r in rows if r['role'] == 'B'], **controls}
    rng, bench = random.Random(3838), []
    for role, items in groups.items():
        items = [r for r in items if lengths[str(r['doc_id'])]['chars'] > 0]
        ordered = sorted(items, key=lambda r: lengths[str(r['doc_id'])]['chars'])
        buckets = {}
        for i, row in enumerate(ordered):
            key = (min(3, i*4//len(ordered)), len(queries[row['qid']]) >= 240)
            buckets.setdefault(key, []).append(row)
        for bucket in buckets.values():
            rng.shuffle(bucket)
        chosen, seen = [], set()
        while len(chosen) < 256:
            moved = False
            for key in sorted(buckets):
                if buckets[key] and len(chosen) < 256:
                    row = buckets[key].pop()
                    chosen.append(row)
                    seen.add(row['pair_id'])
                    moved = True
            assert moved, f'Not enough bench {role}'
        bench.extend(chosen)
    assert len(bench) == len({r['pair_id'] for r in bench}) == 1024
    assert len({r['qid'] for r in bench}) >= 60
    return bench


def recover_windows(rows):
    """New vi37 layer first, otherwise exact latest valid historical provenance owner."""
    owners = dw.read(ROOT/'data/sub/diag37/docwin_max_provenance.json')
    needed = {(r['qid'], str(r['doc_id'])) for r in rows}
    result = {}
    fresh = dw.read(OLD/'docwin_new.json.zst')
    for q, d in needed:
        if fresh.get(q, {}).get(d):
            result[q, d] = ('vi37_new', max(fresh[q][d], key=lambda w: w[2]))
    del fresh
    for name, rel, _ in dw.layer_specs():
        keys = [(q, d) for q, d in needed if (q, d) not in result and owners.get(q, {}).get(d) == name]
        if not keys:
            continue
        layer = dw.read(ROOT/'data/sub'/rel)
        for q, d in keys:
            ws = layer.get(q, {}).get(d, [])
            if ws:
                result[q, d] = (name, max(ws, key=lambda w: w[2]))
        del layer
    return result


def write_parts(dest, prefix, rows, max_tokens=120000):
    files, part, tokens = [], [], 0
    def flush():
        if part:
            p = dest/f'{prefix}.part{len(files):03d}.json'
            dw.write(p, part)
            files.append(p.name)
    for row in rows:
        n = row['n_tokens']
        if part and tokens+n > max_tokens:
            flush()
            part, tokens = [], 0
        part.append(row)
        tokens += n
    flush()
    return files


def stage_export(a):
    from tokenizers import Tokenizer
    import vi37
    tok = Tokenizer.from_file(str(OUT/'tokenizer.json'))
    rows = dw.read(OUT/'selected.json')
    controls = dw.read(OUT/'controls.json')
    queries = {str(r['id']): r['query'] for r in map(json.loads,
              (OLD/'modal_input/queries.jsonl').read_text(encoding='utf-8').splitlines())}
    lengths = dw.read(ROOT/'data/sub/diag37/doc_lengths.json')
    lengths.update(dw.read(OLD/'export_lengths.json'))
    supplemental = ROOT/'data/sub/llmj_survey/supplemental_lengths.json'
    if supplemental.exists():
        lengths.update(dw.read(supplemental))
    bench = stratified_bench(rows, controls, lengths, queries)
    allrows = {r['pair_id']: r for r in rows+bench}
    windows = recover_windows(allrows.values())
    dest = OUT/'export'
    dest.mkdir(parents=True, exist_ok=True)
    wanted = {r['doc_id'] for r in allrows.values()}
    _, texts, textfiles = vi37.export_texts(wanted, dest, 'docs', keep=wanted)
    scores = dw.read(ROOT/'data/sub/diag37/docwin_max_v37.json')
    rendered = {}
    for pid, row in allrows.items():
        text = texts[row['doc_id']]
        owner, window = windows.get((row['qid'], str(row['doc_id'])), (None, None))
        status = 'ready' if text.strip() and window else 'abstain'
        if window:
            assert math.isfinite(window[2]) and abs(window[2]-scores[row['qid']][str(row['doc_id'])]) < 1e-8
        head, win, offsets, full_tokens = snippets(tok, text, window[:2] if window else None)
        prompt = chat(queries[row['qid']], head, win)
        n = len(tok.encode(prompt, add_special_tokens=False).ids)
        assert n < 4096
        rendered[pid] = dict(row, status=status, window_owner=owner,
             window_offset=window[:2] if window else None, excerpt_offsets=offsets,
             text_sha256=sha(text), prompt_sha256=sha(prompt), prompt=prompt,
             n_tokens=n, full_doc_tokens=full_tokens, model=MODEL, revision=REVISION)
    bulk = [rendered[r['pair_id']] for r in rows if rendered[r['pair_id']]['status'] == 'ready']
    basebench = [rendered[r['pair_id']] for r in bench]
    # Empty text remains in benchmark metadata but must never be sent as evidence.
    assert all(r['status'] == 'ready' for r in basebench), 'Bench sampled empty text; inspect rather than impute'
    rng = random.Random(3839)
    repeats = [dict(r, request_id=f'repeat:{i}', repeat_of=r['pair_id'])
               for i, r in enumerate(rng.sample(basebench, 128))]
    original = [dict(r, request_id=f'base:{i}') for i, r in enumerate(basebench)]
    modes = {'bulk': write_parts(dest, 'bulk', bulk),
             'bench': write_parts(dest, 'bench', original+repeats), 'confirm': []}
    dw.write(dest/'bench.json', original+repeats)
    dw.write(dest/'pairs.json', [rendered[r['pair_id']] for r in rows])
    blind, key = [], {}
    shuffled = list(basebench)
    rng.shuffle(shuffled)
    for row in shuffled:
        code = f'J{rng.getrandbits(96):024x}'
        text = texts[row['doc_id']]
        enc = tok.encode(text, add_special_tokens=False)
        end = enc.offsets[min(3000, len(enc.ids))-1][1] if enc.ids else 0
        blind.append(dict(code=code, qid=row['qid'], query=queries[row['qid']],
                          doc_id=row['doc_id'], text=text[:end], truncated=end < len(text)))
        key[code] = {k: v for k, v in row.items() if k != 'prompt'}
    dw.write(dest/'blind.json', blind)
    dw.write(OUT/'private/blind_key.json', key)  # never upload with judge input
    files = [*textfiles, dest/'pairs.json', dest/'bench.json', dest/'blind.json',
             OUT/'tokenizer.json', *[dest/n for ns in modes.values() for n in ns]]
    manifest = dict(model=MODEL, revision=REVISION, vllm=VLLM, modes=modes,
                    files={p.name: fingerprint(p) for p in files},
                    tokenizer='Qwen pinned tokenizer.json; measured, not ceil(chars/3)',
                    total_selected=len(rows), bulk_ready=len(bulk),
                    abstain=len(rows)-len(bulk), bench_unique=1024, repeats=128,
                    blind=1024, bench_queries=len({r['qid'] for r in basebench}),
                    role_counts=dict(Counter(r['role'] for r in rows)),
                    bulk_prompt_tokens=sum(r['n_tokens'] for r in bulk),
                    bench_prompt_tokens=sum(r['n_tokens'] for r in original+repeats),
                    bench_strata=dict(Counter(r['role'] for r in basebench)),
                    long_query_bench=sum(len(queries[r['qid']]) >= 240 for r in basebench),
                    source=fingerprint(OLD/'submission_vi37.json'))
    dw.write(dest/'manifest.json', manifest)
    dw.write(OUT/'export_report.json', manifest)
    print('EXPORT', {k: v for k, v in manifest.items() if k not in ('files', 'modes')}, flush=True)


def checked_merge(expected, outputs):
    wanted = {r['pair_id']: r for r in expected}
    got = {}
    for row in outputs:
        pid = row['pair_id']
        assert pid in wanted and pid not in got, ('unexpected/duplicate', pid)
        ex = wanted[pid]
        assert row['prompt_sha256'] == ex['prompt_sha256'] and row['text_sha256'] == ex['text_sha256']
        assert row['revision'] == REVISION
        parsed = parse_logprobs({i: row.get(f'logp{i}') for i in range(3)}, (0, 1, 2))
        assert parsed['status'] == 'ok', ('missing/nonfinite', pid)
        got[pid] = {**row, **parsed}
    assert set(got) == set(wanted), ('missing pairs', len(set(wanted)-set(got)))
    return got


def stage_merge(a):
    manifest = dw.read(OUT/'export/manifest.json')
    expected = []
    output = []
    for name in manifest['modes']['bulk']:
        assert fingerprint(OUT/'export'/name) == manifest['files'][name]
        expected.extend(dw.read(OUT/'export'/name))
        part = dw.read(OUT/('bulk_out' if getattr(a, 'rule', 'a') == 'b' else 'modal_out')/(name+'.out.json'))
        assert part['input_sha256'] == manifest['files'][name]['sha256']
        output.extend(part['rows'])
    got = checked_merge(expected, output)
    dw.write(OUT/'merged.json', got)
    labels = {role: dict(Counter(str(j['label']) for j in got.values() if j['role'] == role))
              for role in ('A', 'B')}
    dw.write(OUT/'merge_report.json', {'expected': len(expected), 'finite': len(got),
             'coverage': 1., 'labels': labels})


def valid_judgment(j):
    return (j.get('status') == 'ok' and j.get('label') in (0, 1, 2)
            and j.get('score') is not None and math.isfinite(j['score']))


def confirmation_candidates(pairs, got, rule='a'):
    byq = {}
    for r in pairs:
        byq.setdefault(r['qid'], []).append(r)
    wanted = []
    for rows in byq.values():
        aa = [r for r in rows if r['role'] == 'A' and r.get('status') != 'abstain'
              and valid_judgment(got.get(r['pair_id'], {}))
              and got[r['pair_id']]['label'] in ((0, 1) if rule == 'b' else (0,))]
        aa.sort(key=lambda r: (got[r['pair_id']]['score'], r['rank'], r['doc_id']))
        if rule == 'a':
            nb = sum(r['role'] == 'B' and got.get(r['pair_id'], {}).get('label') == 2 for r in rows)
            aa = aa[:min(5, nb)]
        wanted.extend(aa)
    return wanted


def fulltext_prompt(tok, query, text, max_tokens=4095):
    """Longest original character prefix fitting prompt+one decode <=4096 tokens."""
    prompt = chat(query, text, '')
    n = len(tok.encode(prompt, add_special_tokens=False).ids)
    if n <= max_tokens:
        return prompt, n, False
    lo, hi = 0, len(text)
    assert len(tok.encode(chat(query, '', ''), add_special_tokens=False).ids) <= max_tokens
    while lo < hi:
        mid = (lo+hi+1)//2
        if len(tok.encode(chat(query, text[:mid], ''), add_special_tokens=False).ids) <= max_tokens:
            lo = mid
        else:
            hi = mid-1
    prompt = chat(query, text[:lo], '')
    n = len(tok.encode(prompt, add_special_tokens=False).ids)
    assert n <= max_tokens
    return prompt, n, True


def stage_confirm(a):
    """Prepare full-text rechecks of every A=0, never execute them automatically."""
    from tokenizers import Tokenizer
    from rerank_run import load_docs
    tok = Tokenizer.from_file(str(OUT/'tokenizer.json'))
    pairs = dw.read(OUT/'export/pairs.json')
    got = dw.read(OUT/'merged.json')
    rule = getattr(a, 'rule', 'a')
    wanted = confirmation_candidates(pairs, got, rule)
    if rule == 'a':
        assert len(wanted) <= 6000, 'STOP: confirmation exceeds plan reserve; request approval'
    docs = load_docs([str(p) for p in (OUT/'export').glob('docs.part*.jsonl.zst')],
                     wanted={r['doc_id'] for r in wanted})
    query = {str(r['id']): r['query'] for r in map(json.loads,
             (OLD/'modal_input/queries.jsonl').read_text(encoding='utf-8').splitlines())}
    rows, abstain, truncated = [], [], 0
    for r in wanted:
        text = docs[r['doc_id']]
        assert sha(text) == r['text_sha256']
        if rule == 'b':
            prompt, n, cut = fulltext_prompt(tok, query[r['qid']], text)
        else:
            prompt = chat(query[r['qid']], text, '')
            n, cut = len(tok.encode(prompt, add_special_tokens=False).ids), False
        if n >= 4096 or not text.strip():
            abstain.append(r['pair_id'])
            continue
        truncated += int(cut)
        rows.append(dict(r, prompt=prompt, prompt_sha256=sha(prompt), n_tokens=n, truncated=cut,
                         excerpt_prompt_sha256=r['prompt_sha256'], confirm_text_sha256=sha(text)))
    if rule == 'a':
        assert sum(r['n_tokens'] for r in rows) <= 25000000
    manifest = dw.read(OUT/'export/manifest.json')
    names = write_parts(OUT/'export', 'confirm', rows)
    manifest['modes']['confirm'] = names
    manifest['files'].update({n: fingerprint(OUT/'export'/n) for n in names})
    dw.write(OUT/'export/manifest.json', manifest)
    dw.write(OUT/'confirm_report.json', dict(rule=rule, eligible_before=len(wanted),
             eligible=len(rows), truncated=truncated, abstain=abstain,
             tokens=sum(r['n_tokens'] for r in rows), files=names,
             eligible_by_query=dict(Counter(r['qid'] for r in wanted))))


def stage_cmerge(a):
    manifest = dw.read(OUT/'export/manifest.json')
    expected, output = [], []
    for name in manifest['modes']['confirm']:
        expected.extend(dw.read(OUT/'export'/name))
        assert fingerprint(OUT/'export'/name) == manifest['files'][name]
        ob = dw.read(OUT/('confirm_out' if getattr(a, 'rule', 'a') == 'b' else 'modal_out')/(name+'.out.json'))
        assert ob['input_sha256'] == manifest['files'][name]['sha256']
        output.extend(ob['rows'])
    confirmed = checked_merge(expected, output)
    got = dw.read(OUT/'merged.json')
    # Original excerpt logprobs remain the sorting score; confirmation only gates removal.
    for pid, row in confirmed.items():
        got[pid]['confirm_full'] = row['label'] != 2 if getattr(a, 'rule', 'a') == 'b' else row['label'] == 0
        got[pid]['confirm_text_sha256'] = row['text_sha256']
        got[pid]['confirmation'] = {k: row[k] for k in ('label', 'score', 'logp0', 'logp1', 'logp2')}
    dw.write(OUT/'merged.json', got)
    dw.write(OUT/'cmerge_report.json', dict(finite=len(confirmed),
             eligible_after=sum(j['confirm_full'] for j in got.values() if 'confirmation' in j),
             eligible_after_by_query=dict(Counter(j['qid'] for j in got.values()
                                         if j.get('confirm_full') and 'confirmation' in j)),
             labels=dict(Counter(str(j['label']) for j in confirmed.values()))))


def swap_docs(item, candidates, judgments, cap=5, rule='a'):
    assert 0 <= cap <= 5
    docs = list(item['relevant_docs'])
    protected = {int(c['doc_id']) for c in item['relevant_chunks']}
    aa, bb = [], []
    for r in candidates:
        if r.get('status') == 'abstain':
            continue
        j = judgments.get(r['pair_id'], {})
        if not valid_judgment(j):
            continue
        if r['role'] == 'A' and j.get('label') in ((0, 1) if rule == 'b' else (0,)) and r['doc_id'] not in protected:
            # Full-text confirmation is mandatory before any actual removal (plan §5).
            if not j.get('confirm_full') or j.get('confirm_text_sha256') != r.get('text_sha256'):
                continue
            if rule == 'b':
                full = j.get('confirmation', {})
                if not valid_judgment(dict(full, status='ok')) or full['label'] == 2:
                    continue
                j = dict(j, score=full['score'])
            aa.append((r, j))
        if r['role'] == 'B' and j.get('label') == 2 and r['doc_id'] not in docs:
            bb.append((r, j))
    aa.sort(key=lambda t: (t[1]['score'], t[0]['rank'], t[0]['doc_id']))
    bb.sort(key=lambda t: (-t[1]['score'], t[0]['rank'], t[0]['doc_id']))
    exchanges = []
    for (old, _), (new, _) in list(zip(aa, bb))[:cap]:
        assert old['doc_id'] in docs and new['doc_id'] not in docs
        pos = docs.index(old['doc_id'])
        docs[pos] = new['doc_id']  # paired sorted B replaces that sorted A's exact slot
        exchanges.append(dict(position=pos, out=old['doc_id'], incoming=new['doc_id']))
    assert len(docs) == len(set(docs)) == len(item['relevant_docs'])
    assert protected <= set(docs)
    return docs, exchanges


def replace_docs_bytes(raw, submission, newdocs):
    matches = list(re.finditer(rb'"relevant_docs"\s*:\s*\[[^\]]*\]', raw))
    assert len(matches) == len(submission)
    chunks, cursor = [], 0
    for match, item in zip(matches, submission):
        q = str(item['id'])
        chunks.append(raw[cursor:match.start()])
        original = match.group()
        if item['relevant_docs'] == newdocs[q]:
            chunks.append(original)
        else:
            prefix = original[:original.index(b'[')]
            chunks.append(prefix+json.dumps(newdocs[q]).encode('ascii'))
        cursor = match.end()
    chunks.append(raw[cursor:])
    return b''.join(chunks)


def stage_swap(a):
    sub = dw.read(OLD/'submission_vi37.json')
    pairs = dw.read(OUT/'export/pairs.json')
    judgments = {} if a.cap == 0 else dw.read(OUT/'merged.json')
    byq = {}
    for r in pairs:
        byq.setdefault(r['qid'], []).append(r)
    docs, exchanges = {}, {}
    pool = dw.read(ROOT/'data/sub/vi36/in1/pool_v36.json')
    for item in sub:
        q = str(item['id'])
        docs[q], exchanges[q] = swap_docs(item, byq.get(q, []), judgments, a.cap, getattr(a, 'rule', 'a'))
        assert set(docs[q]) <= {int(h[0]) for h in pool[q]}
    name = output_name(a)
    raw = replace_docs_bytes((OLD/'submission_vi37.json').read_bytes(), sub, docs)
    (OUT/f'submission_{name}.json').write_bytes(raw)
    dw.write(OUT/f'exchanges_{name}.json', exchanges)


def stage_build(a):
    import vi29
    name = output_name(a)
    jp, zp = OUT/f'submission_{name}.json', OUT/f'submission_{name}.zip'
    sub, old = dw.read(jp), dw.read(OLD/'submission_vi37.json')
    assert len(sub) == len(old) == 1200
    assert {x['id'] for x in sub} == set(range(1, 1201))
    assert all(x['id'] == y['id'] and x['relevant_chunks'] == y['relevant_chunks'] for x, y in zip(sub, old))
    pool = dw.read(ROOT/'data/sub/vi36/in1/pool_v36.json')
    for item, ref in zip(sub, old):
        ds, baseline = set(item['relevant_docs']), set(ref['relevant_docs'])
        assert len(ds) == len(item['relevant_docs']) == len(ref['relevant_docs'])
        assert len(ds-baseline) == len(baseline-ds) <= 5
        assert ds <= {int(h[0]) for h in pool[str(item['id'])]}
    pattern = rb'("relevant_docs"\s*:\s*)\[[^\]]*\]'
    assert re.sub(pattern, rb'\1[]', jp.read_bytes()) == re.sub(
           pattern, rb'\1[]', (OLD/'submission_vi37.json').read_bytes()), 'Non-doc bytes changed'
    docs_match = sum(x['relevant_docs'] == y['relevant_docs'] for x, y in zip(sub, old))
    if a.cap == 0:
        assert docs_match == 1200 and jp.read_bytes() == (OLD/'submission_vi37.json').read_bytes()
    bad = set(map(int, dw.read(ROOT/'data/sub/diag28/bad_ids.json'))) - set(
          map(int, dw.read(ROOT/'data/sub/diag30/new_ids.json')))
    corpus = vi29.load_corpus_ids(ROOT/'data/sub/diag19/links_corpus_vi_en_zh.parquet')
    clean = vi29.assert_clean(sub, bad, corpus)
    subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate.py')), 'validate', str(jp)], check=True)
    with zipfile.ZipFile(zp, 'w', compression=zipfile.ZIP_LZMA) as z:
        z.write(jp, 'submission.json')
    assert zp.stat().st_size <= 99000000
    rp = OUT/f'report_chain_{name}.json'
    dw.write(rp, {'delta': 55, 'tong_chunk': sum(len(i['relevant_chunks']) for i in sub)})
    code = vi29.core.check_zip(str(jp), str(zp), str(rp), name, max_zip=99000000)
    assert code == 0
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ['submission.json'] and z.testzip() is None
        assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        assert sha(z.read('submission.json')) == fingerprint(jp)['sha256']
    dw.write(OUT/f'build_report_{name}.json', dict(delta=55, docs_match=docs_match,
             chunks_match=1200, chunk_bytes_unchanged=True, validate=0, check_zip=code,
             assert_clean=clean, json=fingerprint(jp), zip=fingerprint(zp)))
    print('BUILD', name, docs_match, 'chunks 1200', code, flush=True)


def output_name(a):
    return 'sanity' if a.cap == 0 else ('vi38b' if getattr(a, 'rule', 'a') == 'b' else 'vi38')


def stage_stats(a):
    pairs = dw.read(OUT/'export/pairs.json')
    manifest = dw.read(OUT/'export/manifest.json')
    name = output_name(a)
    exchanges = dw.read(OUT/f'exchanges_{name}.json')
    rep = dict(selected=len(pairs), roles=dict(Counter(r['role'] for r in pairs)),
               swaps=sum(map(len, exchanges.values())),
               build=dw.read(OUT/f'build_report_{name}.json'),
                bulk_tokens=manifest['bulk_prompt_tokens'], bench_tokens=manifest['bench_prompt_tokens'])
    counts = [len(v) for v in exchanges.values()]
    lookup = {r['pair_id']: r for r in pairs}
    ranks = {side: [lookup[f'{q}:{c[key]}']['rank'] for q, cs in exchanges.items() for c in cs]
             for side, key in [('out', 'out'), ('in', 'incoming')]}
    old = dw.read(OLD/'submission_vi37.json')
    new = dw.read(OUT/f'submission_{name}.json')
    jac = [len(set(x['relevant_docs']) & set(y['relevant_docs'])) /
           len(set(x['relevant_docs']) | set(y['relevant_docs'])) for x, y in zip(old, new)]
    rep.update(queries_with_swap=sum(n > 0 for n in counts), mean_swaps=sum(counts)/1200,
               swaps_distribution=dict(Counter(counts)), rank_distribution={k: dict(Counter(v)) for k, v in ranks.items()},
               rank_summary={k: dict(min=min(v), max=max(v), mean=sum(v)/len(v)) if v else {} for k, v in ranks.items()},
               jaccard_mean=sum(jac)/1200, jaccard_by_query=jac)
    rep['rank_quantiles'] = {k: dw.dist(v) for k, v in ranks.items()}
    dw.write(OUT/f'stats_report_{name}.json', rep)
    print('STATS', rep, flush=True)


def stage_benchstats(a):
    manifest = dw.read(OUT/'export/manifest.json')
    expected = dw.read(OUT/'export/bench.json')
    outputs = []
    for name in manifest['modes']['bench']:
        part = dw.read(OUT/'modal_out'/(name+'.out.json'))
        assert part['input_sha256'] == manifest['files'][name]['sha256']
        outputs.extend(part['rows'])
    byid = {r['request_id']: r for r in outputs}
    assert len(byid) == len(outputs) == len(expected) == 1152
    for r in expected:
        ob = byid[r['request_id']]
        assert ob['prompt_sha256'] == r['prompt_sha256'] and ob['revision'] == REVISION
        parsed = parse_logprobs({i: ob.get(f'logp{i}') for i in range(3)}, (0, 1, 2))
        assert parsed['status'] == 'ok'
        ob.update(parsed)
    originals = {r['pair_id']: byid[r['request_id']] for r in expected if 'repeat_of' not in r}
    repeated = [(originals[r['repeat_of']], byid[r['request_id']]) for r in expected if 'repeat_of' in r]
    agreement = sum(x['label'] == y['label'] for x, y in repeated)/128
    assert agreement >= .99, ('STOP label stability', agreement)
    # A/B exact ordering decisions must not flip under duplicates, including 0/2 eligibility.
    assert all(x['label'] == y['label'] for x, y in repeated if x['role'] in ('A', 'B'))
    sources = {'q8b': dw.read(OLD/'q8b_v37.json'), 'fhyde': dw.read(OLD/'cache_fhyde_v37.json')}
    scores = dw.read(ROOT/'data/sub/diag37/docwin_max_v37.json')
    sources['docwin'] = {q: [[int(d), None, s] for d, s in ds.items()] for q, ds in scores.items()}
    jscores, regions = {}, {}
    for r in originals.values():
        jscores.setdefault(r['qid'], {})[str(r['doc_id'])] = r['score']
        regions.setdefault(r['role'], Counter())[str(r['label'])] += 1
    doclists = {q: list(ds) for q, ds in jscores.items()}
    correlations = {name: dw.correlation_report(doclists, jscores, source) for name, source in sources.items()}
    rep = dict(coverage=1., finite=1152, unique=1024, repeats=128, label_agreement=agreement,
               labels_by_region={k: dict(v) for k, v in regions.items()}, correlations=correlations,
               note='No gold labels: correlation does not establish relevance/F2; blind gate required')
    dw.write(OUT/'bench_report.json', rep)
    print('BENCH', rep, flush=True)


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['select', 'export', 'merge', 'confirm', 'cmerge', 'swap', 'build', 'stats', 'benchstats', 'sanity', 'status'])
    ap.add_argument('--cap', type=int, choices=range(6), default=5)
    ap.add_argument('--rule', choices=['a', 'b'], default='a')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'status':
        lock = ROOT/'locks/heavy_job.lock'
        print('LOCK', lock.read_text() if lock.exists() else 'absent')
        for p in sorted(OUT.glob('*report*.json')):
            print(p.name, p.stat().st_size)
        return
    with dw.owned_lock(ROOT/'locks/heavy_job.lock', f'vi38 {a.stage} pid={os.getpid()}'):
        if a.stage == 'sanity':
            a.cap = 0
            for fn in (stage_swap, stage_build, stage_stats):
                fn(a)
        else:
            globals()['stage_'+a.stage](a)
    print('DONE', a.stage, flush=True)


if __name__ == '__main__':
    main()
