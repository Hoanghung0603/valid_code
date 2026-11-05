"""Exact paragraph-span localization; structural checks only, never a local scorer."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from generate_translation_submission import iter_json_array, sha256_file
from prepare_p2a_chunk_pairs import CONTROL, CONTROL_SHA256, QUERIES, PILOT, digest

ROOT = Path(__file__).resolve().parents[1]
MODEL = 'BAAI/bge-reranker-v2-m3'
REVISION = '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
ANCHOR_ZIP = ROOT / 'data/kaggle_outputs/p2a_chunk_full/p2a_bge_chunk_full_scores.zip'
ANCHOR_HASH = 'df45e378ff334375c819ad9e3dfa283fbb1b99cec0321982701948f3a189779f'
SCHEMA = pa.schema([
    ('pair_id', pa.string()), ('query_id', pa.int64()), ('chunk_ordinal', pa.int64()),
    ('doc_id', pa.int64()), ('candidate_id', pa.int64()), ('start', pa.int64()),
    ('end', pa.int64()), ('parent_sha256', pa.string()), ('text_sha256', pa.string()),
    ('query_text', pa.string()), ('passage_text', pa.string()), ('kind', pa.string()),
])


def paragraphs(text):
    """Only fully newline-bounded internal paragraphs; clipped edge blocks excluded."""
    lines = list(re.finditer(r'[^\r\n]+', text))
    spans = []
    for match in lines:
        start, end = match.span()
        if start == 0 or end == len(text):
            continue
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end-1].isspace():
            end -= 1
        if start < end:
            spans.append((start, end))
    return spans


def candidate_spans(text, anchor_start, anchor_end):
    """Up to 8 adjacent paragraph groups near the existing strongest window.

    Character lengths only remove tiny/near-identical options, not define borders.
    Actual model token lengths are checked by the worker without truncation.
    """
    units = paragraphs(text)
    middle = (anchor_start + anchor_end) / 2
    options = set()
    for i in range(len(units)):
        for count in (1, 2, 3):
            if i + count > len(units):
                continue
            start, end = units[i][0], units[i+count-1][1]
            if end-start < 120 or end-start >= 0.95 * len(text):
                continue
            if end <= anchor_start or start >= anchor_end:
                continue
            options.add((start, end))
    ordered = sorted(options, key=lambda p: (0 if p[0] <= middle <= p[1] else 1,
                     abs((p[0]+p[1])/2-middle), -(p[1]-p[0]), p[0]))
    return [(0, len(text))] + ordered[:8]


def choose_candidate(group):
    """Predeclared exploratory gate, not a calibrated relevance probability."""
    original = group[group.candidate_id == 0].iloc[0]
    eligible = group[(group.candidate_id != 0) & (group.status == 'ok')]
    if eligible.empty:
        return original
    best = eligible.sort_values(['raw_score', 'span_chars', 'candidate_id'],
                                ascending=[False, False, True]).iloc[0]
    # Low-score/tie cases fall back. Compare with whole parent only when fully scored.
    if best.raw_score < 1.0:
        return original
    if original.status == 'ok' and best.raw_score < original.raw_score + 0.5:
        return original
    others = eligible[eligible.candidate_id != best.candidate_id]
    if not others.empty and (others.raw_score == best.raw_score).any():
        return original
    return best


def prepare(scope, out):
    if out.exists():
        raise FileExistsError(out)
    if sha256_file(CONTROL) != CONTROL_SHA256 or sha256_file(ANCHOR_ZIP) != ANCHOR_HASH:
        raise RuntimeError('Control or audited anchor artifact changed')
    queries = pd.read_parquet(QUERIES).set_index('id')['query'].to_dict()
    wanted = set(pd.read_parquet(PILOT).id) if scope == 'pilot' else set(queries)
    if len(queries) != 1200 or len(wanted) != (120 if scope == 'pilot' else 1200):
        raise RuntimeError('Wrong query coverage')
    with zipfile.ZipFile(ANCHOR_ZIP) as z:
        anchors = pd.read_parquet(io.BytesIO(z.read('p2a_pair_scores.parquet')))
    anchors = anchors.sort_values(['normalized_score', 'window_start'], ascending=[False, True])
    anchors = anchors.drop_duplicates(['query_id', 'chunk_ordinal']).set_index(['query_id', 'chunk_ordinal'])
    out.mkdir(parents=True)
    pair_path = out / 'p2b_pairs.parquet'
    counts = Counter()
    seen = set()
    with pq.ParquetWriter(pair_path, SCHEMA, compression='zstd') as writer, zipfile.ZipFile(CONTROL) as z:
        with z.open('submission.json') as binary, io.TextIOWrapper(binary, encoding='utf-8') as stream:
            for row in iter_json_array(stream):
                qid = row['id']
                if qid not in wanted:
                    continue
                if qid in seen:
                    raise RuntimeError('Duplicate query')
                seen.add(qid)
                records = []
                for ordinal, chunk in enumerate(row['relevant_chunks']):
                    text = chunk['chunk_text']
                    anchor = anchors.loc[(qid, ordinal)]
                    if anchor.chunk_sha256 != digest(text) or anchor.doc_id != chunk['doc_id']:
                        raise RuntimeError('Anchor lineage mismatch')
                    spans = candidate_spans(text, anchor.window_start, anchor.window_end)
                    counts['chunks'] += 1
                    counts['chunks_with_alternatives'] += len(spans) > 1
                    for cid, (start, end) in enumerate(spans):
                        records.append(dict(pair_id=f'{qid}:{ordinal}:{cid}', query_id=qid,
                            chunk_ordinal=ordinal, doc_id=chunk['doc_id'], candidate_id=cid,
                            start=start, end=end, parent_sha256=digest(text), text_sha256=digest(text[start:end]),
                            query_text=queries[qid], passage_text=text[start:end],
                            kind='parent' if cid == 0 else 'paragraph_group'))
                counts['pairs'] += len(records)
                writer.write_table(pa.Table.from_pylist(records, schema=SCHEMA))
                if len(seen) % 20 == 0:
                    print(dict(queries=len(seen), **counts), flush=True)
    if seen != wanted:
        raise RuntimeError('Missing queries')
    manifest = dict(schema='p2b_localization_v1', scope=scope, control_sha256=CONTROL_SHA256,
                    anchor_sha256=ANCHOR_HASH, queries_sha256=sha256_file(QUERIES),
                    pilot_sha256=sha256_file(PILOT) if scope == 'pilot' else None,
                    pair_sha256=sha256_file(pair_path), queries=len(seen), **counts,
                    model=MODEL, revision=REVISION, max_tokens=1024,
                    policy=dict(min_raw=1.0, parent_margin=0.5, max_alternatives=8,
                                note='Heuristic, not calibrated; original-query only; no local qrels.'))
    (out / 'p2b_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print(json.dumps(manifest, indent=2))


def validate_scores(pack, result):
    manifest = json.loads((pack / 'p2b_manifest.json').read_text(encoding='utf-8'))
    if (manifest['schema'] != 'p2b_localization_v1' or manifest['revision'] != REVISION
            or manifest['control_sha256'] != CONTROL_SHA256 or manifest['model'] != MODEL
            or manifest['max_tokens'] != 1024
            or manifest['policy']['min_raw'] != 1.0 or manifest['policy']['parent_margin'] != 0.5):
        raise RuntimeError('Unsupported run policy/provenance')
    if sha256_file(pack / 'p2b_pairs.parquet') != manifest['pair_sha256']:
        raise RuntimeError('Input changed')
    with zipfile.ZipFile(result) as z:
        if set(z.namelist()) != {'scores.parquet', 'run.json'} or z.testzip() is not None:
            raise RuntimeError('Score ZIP invalid')
        run = json.loads(z.read('run.json'))
        data = z.read('scores.parquet')
        if run['input'] != manifest or hashlib.sha256(data).hexdigest() != run['score_sha256']:
            raise RuntimeError('Score provenance mismatch')
        if run.get('truncation') is not False or run.get('worker_sha256') != sha256_file(ROOT/'scripts/p2b_score_worker.py'):
            raise RuntimeError('Worker code changed or truncation enabled')
        identities = run.get('workers', [])
        if len(identities) != 2 or {w['shard'] for w in identities} != {0, 1}:
            raise RuntimeError('Missing GPU worker identities')
        for identity in identities:
            if (identity['input_sha256'] != manifest['pair_sha256'] or identity['revision'] != REVISION
                    or identity['model'] != MODEL or identity['max_tokens'] != 1024
                    or identity['truncation'] is not False or identity['worker_sha256'] != run['worker_sha256']):
                raise RuntimeError('Worker identity mismatch')
        scores = pd.read_parquet(io.BytesIO(data))
    pairs = pd.read_parquet(pack / 'p2b_pairs.parquet')
    if pairs.pair_id.duplicated().any() or len(pairs) != manifest['pairs'] or pairs.query_id.nunique() != manifest['queries']:
        raise RuntimeError('Input coverage mismatch')
    if scores.pair_id.duplicated().any() or set(scores.pair_id) != set(pairs.pair_id):
        raise RuntimeError('Score coverage mismatch')
    if set(scores.revision) != {REVISION} or not set(scores.status) <= {'ok', 'too_long'}:
        raise RuntimeError('Invalid revision/status')
    for s in scores.itertuples():
        if s.status == 'ok' and (not math.isfinite(s.raw_score) or s.input_tokens > 1024):
            raise RuntimeError('Invalid full-text score')
        if s.status == 'too_long' and s.input_tokens <= 1024:
            raise RuntimeError('Incorrect overflow status')
        if s.status == 'too_long' and not pd.isna(s.raw_score):
            raise RuntimeError('Oversize input must not receive a score')
    pairs = pairs.merge(scores, on='pair_id', validate='one_to_one')
    expected_shards = {qid: i % 2 for i,qid in enumerate(sorted(pairs.query_id.unique()))}
    if not (pairs.shard == pairs.query_id.map(expected_shards)).all():
        raise RuntimeError('GPU partition lineage mismatch')
    pairs['span_chars'] = pairs.end - pairs.start
    return manifest, pairs


def verify_group(text, doc_id, group):
    original = group[group.candidate_id == 0]
    if len(original) != 1 or original.iloc[0].start != 0 or original.iloc[0].end != len(text):
        raise RuntimeError('Missing/invalid whole-parent fallback')
    boundaries = paragraphs(text)
    starts, ends = {a for a,b in boundaries}, {b for a,b in boundaries}
    for candidate in group.itertuples():
        if candidate.doc_id != doc_id or candidate.parent_sha256 != digest(text):
            raise RuntimeError('Parent mapping mismatch')
        if not 0 <= candidate.start < candidate.end <= len(text):
            raise RuntimeError('Invalid span offsets')
        if candidate.candidate_id and (candidate.start not in starts or candidate.end not in ends):
            raise RuntimeError('Non-paragraph boundary')
        if text[candidate.start:candidate.end] != candidate.passage_text or digest(candidate.passage_text) != candidate.text_sha256:
            raise RuntimeError('Source text/hash mismatch')


def source_audit(manifest, pairs):
    if sha256_file(CONTROL) != CONTROL_SHA256 or sha256_file(QUERIES) != manifest['queries_sha256']:
        raise RuntimeError('Control/official query changed')
    queries = pd.read_parquet(QUERIES).set_index('id')['query'].to_dict()
    grouped = {qid: group for qid, group in pairs.groupby('query_id')}
    observed = set()
    chunks = 0
    with zipfile.ZipFile(CONTROL) as z:
        with z.open('submission.json') as binary, io.TextIOWrapper(binary, encoding='utf-8') as stream:
            for row in iter_json_array(stream):
                qid = row['id']
                if qid not in grouped:
                    continue
                frame = grouped[qid]
                if set(frame.query_text) != {queries[qid]}:
                    raise RuntimeError('Query changed')
                groups = {int(ordinal): group for ordinal,group in frame.groupby('chunk_ordinal')}
                if set(groups) != set(range(len(row['relevant_chunks']))):
                    raise RuntimeError('Chunk coverage differs')
                for ordinal, chunk in enumerate(row['relevant_chunks']):
                    verify_group(chunk['chunk_text'], chunk['doc_id'], groups[ordinal])
                    chunks += 1
                observed.add(qid)
    if observed != set(grouped) or chunks != manifest['chunks']:
        raise RuntimeError('Source audit incomplete')
    return dict(queries=len(observed), chunks=chunks, pairs=len(pairs), exact_source=True,
                statuses=pairs.groupby(['kind','status']).size().to_dict())


def localize_chunks(original, groups):
    replacements = []
    for ordinal, chunk in enumerate(original):
        group = groups[ordinal]
        verify_group(chunk['chunk_text'], chunk['doc_id'], group)
        chosen = choose_candidate(group)
        replacements.append(dict(doc_id=chunk['doc_id'], chunk_text=chosen.passage_text))
    keys = Counter((c['doc_id'], c['chunk_text']) for c in replacements)
    collisions = {doc for (doc, text), count in keys.items() if count > 1}
    # Roll back all slots of affected docs, including pre-existing duplicates.
    return [old if new['doc_id'] in collisions else new
            for old,new in zip(original, replacements, strict=True)]


def build(pack, result, output):
    """Audit exact source/policy and package only full-coverage submissions."""
    manifest, pairs = validate_scores(pack, result)
    if manifest['scope'] != 'full' or manifest['queries'] != 1200:
        raise RuntimeError('Pilot cannot produce a BTC submission')
    if output.exists() or sha256_file(CONTROL) != manifest['control_sha256']:
        raise RuntimeError('Output exists or control changed')
    source_audit(manifest, pairs)
    groups = {key: group for key, group in pairs.groupby(['query_id', 'chunk_ordinal'])}
    seen = set()
    stats = Counter()
    json_path = output.with_suffix('.json')
    if json_path.exists():
        raise FileExistsError(json_path)
    with zipfile.ZipFile(CONTROL) as z, json_path.open('x', encoding='utf-8') as dest:
        dest.write('[')
        with z.open('submission.json') as binary, io.TextIOWrapper(binary, encoding='utf-8') as stream:
            for row in iter_json_array(stream):
                qid = row['id']
                original = row['relevant_chunks']
                local_groups = {ordinal: groups[(qid, ordinal)] for ordinal in range(len(original))}
                replacements = localize_chunks(original, local_groups)
                seen.update((qid,ordinal) for ordinal in local_groups)
                assert Counter(c['doc_id'] for c in replacements) == Counter(c['doc_id'] for c in original)
                stats['changed_chunks'] += sum(a != b for a, b in zip(original, replacements, strict=True))
                stats['chunks'] += len(original)
                row['relevant_chunks'] = replacements
                if stats['queries']:
                    dest.write(',')
                json.dump(row, dest, ensure_ascii=False, separators=(',', ':'))
                stats['queries'] += 1
        dest.write(']')
    if seen != set(groups) or stats['queries'] != 1200 or stats['changed_chunks'] == 0:
        raise RuntimeError('Coverage failed or no-op')
    from build_p1_verified_docs_submission import package_lzma64
    staged = package_lzma64(json_path, output)
    if staged.stat().st_size >= 100_000_000:
        raise RuntimeError('ZIP exceeds 100 MB; preserve staging for diagnostics, do not submit')
    with zipfile.ZipFile(staged) as z:
        if z.namelist() != ['submission.json'] or z.testzip() is not None:
            raise RuntimeError('ZIP CRC/layout failure')
        with zipfile.ZipFile(CONTROL) as control:
            with control.open('submission.json') as a, z.open('submission.json') as b:
                aa, bb = io.TextIOWrapper(a, encoding='utf-8'), io.TextIOWrapper(b, encoding='utf-8')
                validated = 0
                for old,new in zip(iter_json_array(aa), iter_json_array(bb), strict=True):
                    if old['id'] != new['id'] or old['relevant_docs'] != new['relevant_docs']:
                        raise RuntimeError('Packaged query/docs changed')
                    for before,after in zip(old['relevant_chunks'],new['relevant_chunks'],strict=True):
                        if before['doc_id'] != after['doc_id'] or not after['chunk_text'] or after['chunk_text'] not in before['chunk_text']:
                            raise RuntimeError('Packaged chunk is not an exact same-owner source span')
                    validated += 1
                if validated != 1200:
                    raise RuntimeError('Wrong packaged query count')
    staged.rename(output)
    report = dict(**stats, input=manifest, scores_sha256=sha256_file(result),
                  zip_sha256=sha256_file(output), zip_bytes=output.stat().st_size,
                  status='structural_checks_passed_btc_score_pending')
    output.with_suffix('.manifest.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--scope', choices=['pilot', 'full'], required=True)
    prep.add_argument('--output', type=Path, required=True)
    score = sub.add_parser('audit')
    score.add_argument('--pack', type=Path, required=True)
    score.add_argument('--result', type=Path, required=True)
    candidate = sub.add_parser('build')
    candidate.add_argument('--pack', type=Path, required=True)
    candidate.add_argument('--result', type=Path, required=True)
    candidate.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args.scope, args.output)
    elif args.command == 'build':
        build(args.pack, args.result, args.output)
    else:
        manifest, pairs = validate_scores(args.pack, args.result)
        report = source_audit(manifest, pairs)
        report['statuses'] = {f'{k[0]}:{k[1]}':v for k,v in report['statuses'].items()}
        report.update(scope=manifest['scope'], result_sha256=sha256_file(args.result), quality_evaluation=False)
        target = args.result.with_suffix('.audit.json')
        if target.exists():
            raise FileExistsError(target)
        target.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
