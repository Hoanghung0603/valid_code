"""CPU-only CX expansion; historical inputs are immutable, no submission actions."""
import argparse
from collections import Counter, defaultdict
import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import zstandard as zstd

from vi41c import indexed, load_zip
from vi45 import cid
from vi48 import distribution, fingerprint
from reorder_zip import write_reordered_zip

ROOT = Path(__import__("os").environ.get("WORKDIR", "<WORKDIR>") + '/r2ai_deliver')
INPUT = ROOT/'cand/vi57_inputs'
HIST = INPUT/'replay/inputs'
CORPUS = ROOT/'internal/data_verified/r2ai_corpus_vi_clean'
LIMIT = 99_000_000


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def iterz(path):
    with zstd.open(path, 'rt', encoding='utf-8') as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def k(chunk):
    return str(chunk['doc_id']), chunk['chunk_text']


def select_row(base, candidates, threshold=0):
    docs = set(map(str, base['relevant_docs']))
    seen = set(map(k, base['relevant_chunks']))
    if len(seen) != len(base['relevant_chunks']):
        raise ValueError('duplicate baseline chunks')
    selected = {}
    for c in candidates:
        s = float(c['score'])
        if not math.isfinite(s):
            raise ValueError('non-finite logit')
        key = k(c)
        if s > threshold and key[0] in docs and key not in seen:
            previous = selected.get(key)
            if previous is None or s > previous['score']:
                selected[key] = c
    return sorted(selected.values(), key=lambda c: (-c['score'], str(c['doc_id']), c['chunk_text']))


def make_rows(base, additions, threshold):
    return [{**row, 'relevant_chunks': row['relevant_chunks'] + [
        {'doc_id': c['doc_id'], 'chunk_text': c['chunk_text']}
        for c in additions.get(str(row['id']), ()) if c['score'] > threshold]}
        for row in base]


def fit_budget(base, additions, measure, limit=LIMIT):
    """Actual ZIP bytes; binary search fixed unique global logits if needed."""
    attempts = []
    def try_threshold(t):
        rows = make_rows(base, additions, t)
        info = measure(rows)
        attempts.append({'threshold': t, **info})
        return rows, info['zip_bytes'] <= limit
    rows, ok = try_threshold(0)
    if ok:
        return rows, 0, attempts
    values = sorted({c['score'] for cs in additions.values() for c in cs})
    if not values: raise ValueError('baseline exceeds limit')
    lo, hi = 0, len(values)-1
    # Measure the empty-suffix upper bound first; at most eight nonzero probes.
    for step in range(8):
        if lo > hi: break
        mid = hi if step == 0 else (lo+hi)//2
        rows, ok = try_threshold(values[mid])
        if ok:
            hi = mid-1
        else:
            lo = mid+1
    fitting = [a['threshold'] for a in attempts if a['zip_bytes'] <= limit]
    if not fitting:
        raise ValueError('baseline exceeds limit')
    best = min(fitting)
    rows, ok = try_threshold(best)
    if not ok:
        raise ValueError('ZIP size not repeatable')
    return rows, best, attempts


def source_parts_vi45():
    source = HIST/'b660/vi45'
    prep = read(source/'prepare_vi44_report.json')
    files = sorted((source/'modal_input').glob('*.jsonl.zst'))
    if len(files) != 20:
        raise ValueError('vi45 input parts incomplete')
    for p in files:
        if fingerprint(p) != prep['files'][p.name]:
            raise ValueError(('vi45 input SHA', str(p)))
        scores = read(source/'outputs'/p.name.replace('.jsonl.zst', '.logits.json'))
        byid = {r['cid']: r for r in scores}
        if len(byid) != len(scores): raise ValueError('duplicate vi45 id')
        count = 0
        for r in iterz(p):
            score = byid.pop(r['cid'])
            if (str(r['qid']), str(r['doc_id'])) != (str(score['qid']), str(score['doc_id'])):
                raise ValueError('vi45 id mismatch')
            yield r, score['logit8c'], 'vi45'
            count += 1
        if byid or count != prep['counts'][p.name]: raise ValueError('vi45 count mismatch')


def source_parts_vi50():
    prep = read(INPUT/'vi50/prepare_report.json')
    scorefiles = sorted((HIST/'modal').glob('*/r2ai-runner/out/vi50/vi50/bulk/*/*.logits.json'))
    if len(scorefiles) != 16: raise ValueError('vi50 score parts incomplete')
    scoremap = {p.name: p for p in scorefiles}
    for name, meta in prep['files'].items():
        p = INPUT/'vi50/modal_input'/name
        if fingerprint(p) != meta: raise ValueError(('vi50 input SHA', name))
        scores = read(scoremap[name.replace('.jsonl.zst', '.logits.json')])
        byid = {r['cid']: r for r in scores}
        if len(scores) != len(byid): raise ValueError('duplicate vi50 id')
        count = 0
        for r in iterz(p):
            s = byid.pop(r['cid'])
            if (str(r['qid']), str(r['doc_id'])) != (str(s['qid']), str(s['doc_id'])):
                raise ValueError('vi50 id mismatch')
            yield r, s['logit8c'], 'vi50'
            count += 1
        if byid or count != prep['counts'][name]: raise ValueError('vi50 count mismatch')


def source_parts_vi56(root, label):
    prep = read(root/'prepare_report.json') if (root/'prepare_report.json').exists() else None
    parts = sorted((root/'in').glob('qwen.part*.jsonl.zst'))
    if not parts: raise ValueError((label, 'no qwen parts'))
    for p in parts:
        meta = prep['parts'][p.name] if prep else None
        if meta and fingerprint(p) != {key: meta[key] for key in ('sha256', 'bytes')}:
            raise ValueError((label, 'input SHA', p.name))
        found = list((root/'out').glob('qwen*/'+p.name+'.scores.json'))
        if len(found) != 1: raise ValueError((label, 'missing or duplicate score', p.name))
        sp = found[0]
        complete = read(sp.with_suffix('.complete.json'))
        if complete['input'] != fingerprint(p) or complete['output'] != fingerprint(sp):
            raise ValueError((label, 'completion SHA', p.name))
        scores = read(sp)
        count = 0
        for count, (r, s) in enumerate(zip(iterz(p), scores, strict=True), 1):
            if any(str(r[key]) != str(s[key]) for key in ('identity', 'qid', 'doc_id', 'kind')):
                raise ValueError((label, 'identity mismatch'))
            if r['kind'] == 'd8w':
                r['cid'] = r['identity']
                yield r, s['score'], label
        if count != complete['pairs'] or (meta and count != meta['count']): raise ValueError((label, 'count mismatch', p.name))


def verbatim(rows):
    pending = defaultdict(set)
    docs = set()
    for row in rows:
        docs.update(map(int, row['relevant_docs']))
        for c in row['relevant_chunks']:
            pending[int(c['doc_id'])].add(c['chunk_text'])
    # IDs without a body are still members of the verified VI corpus.
    remaining_docs = set(docs) - set(map(int, read(CORPUS/'ids_without_text.json')))
    manifest = read(CORPUS/'MANIFEST.json')['files']
    failures = 0
    # Historical exports are alternative authoritative original bodies.
    files = list(sorted(CORPUS.glob('corpus.part*.jsonl.zst')))
    files += list(sorted((HIST/'b660/vi45').glob('docs_verify_vi45.part*.jsonl.zst')))
    for p in files:
        if p.parent == CORPUS and fingerprint(p) != manifest[p.name]:
            raise ValueError(('corpus SHA', str(p)))
        for r in iterz(p):
            doc = int(r['id'])
            if p.parent == CORPUS:
                remaining_docs.discard(doc)
            if doc in pending:
                missing = {text for text in pending[doc] if text not in r['text']}
                if missing: pending[doc] = missing
                else: del pending[doc]
    if pending:
        raise ValueError(('verbatim', len(pending), sum(map(len, pending.values())), list(pending)[:10]))
    return {'verbatim_bad': failures, 'docs_outside_vi_clean': len(remaining_docs),
            'docs_outside_vi_clean_ids': sorted(remaining_docs)}


def build(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    zp = out/f'submission_{args.name}.zip'
    if zp.exists(): raise FileExistsError(zp)
    base = sorted(load_zip(args.base), key=lambda r: r['id'])
    base_index = indexed(base)
    if set(base_index) != set(range(1,1201)): raise ValueError('base IDs')
    # Sanity is mandatory for vi53 regardless of selected base.
    vi53 = base if args.name == 'vi57' else sorted(load_zip(ROOT/'submissions/submission_vi53.zip'), key=lambda r:r['id'])
    vi45 = indexed(load_zip(HIST/'b660/vi45/submission_vi45.zip'))
    needed = set()
    for row in vi53:
        prefix = vi45[row['id']]['relevant_chunks']
        if row['relevant_chunks'][:len(prefix)] != prefix:
            raise ValueError(('STOP sanity vi45 prefix', row['id']))
        for c in row['relevant_chunks'][len(prefix):]:
            needed.add(cid(row['id'], c))
    del vi45
    if vi53 is not base: del vi53
    gc.collect()
    docs = {str(r['id']): set(map(str, r['relevant_docs'])) for r in base}
    existing = {str(r['id']): set(map(k, r['relevant_chunks'])) for r in base}
    selected = defaultdict(dict)
    possible_reused = {}
    all_scores = {}
    first_scores = {}
    duplicate_ranges = {}
    counts = Counter()
    duplicate_count = 0
    max_diff = 0.0
    source_sets = defaultdict(set)
    sources = [source_parts_vi45(), source_parts_vi50(), source_parts_vi56(HIST/'modal/<MODAL_PROFILE_D>/r2ai-vi56-data', 'vi56')]
    if args.extra: sources.append(source_parts_vi56(args.extra, 'extra'))
    for source in sources:
        for r, value, label in source:
            q, d, ident, text = str(r['qid']), str(r['doc_id']), r['cid'], r['text']
            value = float(value)
            c = {'doc_id': int(d), 'chunk_text': text}
            if not math.isfinite(value) or cid(q,c) != ident: raise ValueError(('bad cid or score', label, ident))
            counts[label] += 1
            if label == 'vi45': first_scores[ident] = (q,d,value)
            if label == 'vi45' and value <= 0 and d in docs.get(q, set()) and (d,text) not in existing.get(q,set()):
                possible_reused[ident] = (q,c)
            old = all_scores.get(ident)
            if old is not None:
                low, high = duplicate_ranges.get(ident, (old,old))
                duplicate_ranges[ident] = (min(low,value), max(high,value))
                all_scores[ident] = max(old, value)
            else: all_scores[ident] = value
            source_sets[ident].add(label)
            if value > 0: needed.discard(ident)
            if value <= 0 or d not in docs.get(q, set()) or (d,text) in existing.get(q,set()): continue
            prev = selected[q].get(ident)
            if prev is None or value > prev['score']:
                selected[q][ident] = {**c, 'score': value, 'cid': ident, 'winner': label}
    if counts['vi45'] != 317601 or counts['vi50'] != 264947 or counts['vi56'] != 57513:
        raise ValueError(('source count mismatch', dict(counts)))
    # Reused scores are separate observations and must participate in max and diagnostics.
    reused = read_z_json(HIST/'b660/vi50/reused_logits.json.zst')
    for r in reused:
        ident = r['cid']
        if ident not in first_scores or first_scores[ident][:2] != (str(r['qid']),str(r['doc_id'])):
            raise ValueError(('vi50 reused missing', ident))
        value = float(r['logit8c'])
        if not math.isfinite(value): raise ValueError('nonfinite reused logit')
        old = all_scores[ident]
        low, high = duplicate_ranges.get(ident, (old,old))
        duplicate_ranges[ident] = (min(low,value), max(high,value))
        all_scores[ident] = max(old,value)
        q = str(r['qid'])
        if ident in selected[q] and value > selected[q][ident]['score']:
            selected[q][ident]['score'] = value
            selected[q][ident]['winner'] = 'vi50'
        if value > 0:
            needed.discard(ident)
            if ident not in selected[q] and ident in possible_reused:
                _, chunk = possible_reused[ident]
                selected[q][ident] = {**chunk,'score':value,'cid':ident,'winner':'vi50'}
        source_sets[ident].add('vi50')
    if len(reused) != 274261: raise ValueError('vi50 reused count')
    counts['vi50_reused'] = len(reused)
    differences = [high-low for low,high in duplicate_ranges.values()]
    duplicate_count = len(differences)
    max_diff = max(differences,default=0.0)
    over = sum(x > 1e-3 for x in differences)
    crossing = sum(low <= 0 < high for low,high in duplicate_ranges.values())
    dupe_report = {'pairs':duplicate_count,'max_delta':max_diff,'p99_delta':distribution(differences)['p99'],
                   'over_1e-3':over,'sign_crossings':crossing}
    print('DUPLICATES '+json.dumps(dupe_report),flush=True)
    if duplicate_count and over/duplicate_count > .01:
        raise ValueError(('STOP duplicate mismatches >1%',dupe_report))
    if needed: raise ValueError(('STOP sanity suffix no positive score',len(needed)))
    del reused, all_scores, first_scores, duplicate_ranges, differences, possible_reused
    additions = {q: sorted(items.values(), key=lambda c:(-c['score'],str(c['doc_id']),c['chunk_text'])) for q,items in selected.items()}
    print('SANITY_PASS '+json.dumps({'suffix_missing':0,'sources':dict(counts),'duplicate_count':duplicate_count,'max_delta':max_diff}),flush=True)
    # All possible final chunks and all docs must pass before the first ZIP.
    candidate_rows = make_rows(base, additions, 0)
    from vi29 import assert_clean, load_corpus_ids
    source = HIST/'b660/vi45'
    bad = set(map(int,read(source/'bad_ids.json'))) - set(map(int,read(source/'new_ids.json')))
    clean = assert_clean(candidate_rows, bad, load_corpus_ids(source/'links_corpus_vi_en_zh.parquet'))
    checks = verbatim(candidate_rows)
    print('VERBATIM_PASS',flush=True)
    del candidate_rows
    rows, threshold, attempts = fit_budget(base, additions, lambda data:write_reordered_zip(data,zp))
    actual = indexed(rows)
    for q, row in actual.items():
        old = base_index[q]
        if row['relevant_docs'] != old['relevant_docs'] or row['relevant_chunks'][:len(old['relevant_chunks'])] != old['relevant_chunks']:
            raise ValueError(('prefix or docs changed',q))
        keys = list(map(k,row['relevant_chunks']))
        if len(keys)!=len(set(keys)) or not {d for d,t in keys} <= set(map(str,row['relevant_docs'])):
            raise ValueError(('duplicate or outside doclist',q))
    kept = [c for cs in additions.values() for c in cs if c['score'] > threshold]
    perq = [sum(c['score'] > threshold for c in additions.get(str(r['id']),())) for r in base]
    attribution = {s:sum(s in source_sets[c['cid']] for c in kept) for s in ('vi45','vi50','vi56','extra')}
    winners = Counter(c['winner'] for c in kept)
    from vi41c import digest
    expected_hashes = {r['id']:digest(r) for r in rows}
    with zipfile.ZipFile(zp) as z:
        if z.namelist()!=['submission.json'] or z.infolist()[0].compress_type != zipfile.ZIP_LZMA or z.testzip() is not None:
            raise ValueError('ZIP format or CRC')
        jp = out/f'submission_{args.name}.json'
        with z.open('submission.json') as src, jp.open('xb') as dst: shutil.copyfileobj(src,dst)
    # Validate the exact extracted final artifact, and keep the JSON for audit.
    del rows, actual, base, base_index, additions, selected
    gc.collect()
    final = load_zip(zp)
    if {r['id']:digest(r) for r in final} != expected_hashes:
        raise ValueError('ZIP content differs from intended rows')
    del final
    gc.collect()
    subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),'validate',str(jp)],check=True)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'check-zip','--out',str(out),'--name',args.name],check=True)
    report = {'name':args.name, 'source_counts':dict(counts),'duplicate_statistics':dupe_report,
              'sanity_vi53_prefix':1200,'sanity_suffix_missing':0,'threshold':threshold,
              'appended':len(kept),'per_query':distribution(perq),'zero_queries':perq.count(0),
              'logits':distribution([c['score'] for c in kept]),'source_membership':attribution,'winner_source':dict(winners),
              'attempts':attempts,'zip':fingerprint(zp),'validate':0,'check_zip':0,'assert_clean':clean,'crc':0,'docs_equal':1200,'prefix_equal':1200,**checks}
    (out/f'report_{args.name}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    jp.unlink()
    print('DONE '+json.dumps(report,ensure_ascii=False),flush=True)


def read_z_json(path):
    with zstd.open(path,'rt',encoding='utf-8') as f: return json.load(f)


def main():
    ap=argparse.ArgumentParser()
    sp=ap.add_subparsers(dest='command',required=True)
    b=sp.add_parser('build')
    b.add_argument('--base',type=Path,required=True)
    b.add_argument('--name',choices=('vi57','vi58','vi59'),required=True)
    b.add_argument('--out',type=Path,required=True)
    b.add_argument('--extra',type=Path)
    check = sp.add_parser('check-zip')
    check.add_argument('--out',type=Path,required=True)
    check.add_argument('--name',choices=('vi57','vi58','vi59'),required=True)
    args=ap.parse_args()
    if args.command == 'check-zip':
        from vi23 import check_zip
        raise SystemExit(check_zip(str(args.out/f'submission_{args.name}.json'),str(args.out/f'submission_{args.name}.zip'),str(args.out/'check_zip.json'),args.name,max_zip=LIMIT))
    build(args)


if __name__=='__main__': main()
