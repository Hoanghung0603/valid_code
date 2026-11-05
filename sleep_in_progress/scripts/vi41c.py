"""Build vi41c from vi37 chunks and vi41b docs, then audit it locally."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import shutil
import zipfile


def key(doc):
    return str(doc)


def merge_row(reference, candidate):
    """Keep candidate order, drop its last unprotected docs, append missing reference docs."""
    if reference['id'] != candidate['id']:
        raise ValueError('different ids')
    docs = candidate['relevant_docs']
    refdocs = reference['relevant_docs']
    for name, values in [('vi37', refdocs), ('vi41b', docs)]:
        if len({key(d) for d in values}) != len(values):
            raise ValueError(f'duplicate doc in {name}, id={reference["id"]}')
    protected = {key(c['doc_id']) for c in reference['relevant_chunks']}
    refkeys = {key(d) for d in refdocs}
    if not protected <= refkeys:
        raise ValueError(f'vi37 chunk outside docs, id={reference["id"]}')
    docset = {key(d) for d in docs}
    missing = [d for d in refdocs if key(d) in protected and key(d) not in docset]
    remaining = len(missing)
    removed = set()
    for i in range(len(docs) - 1, -1, -1):
        if remaining and key(docs[i]) not in protected:
            removed.add(i)
            remaining -= 1
    result = [d for i, d in enumerate(docs) if i not in removed] + missing
    return {'id': reference['id'], 'relevant_docs': result,
            'relevant_chunks': reference['relevant_chunks']}, len(missing), remaining


def load_zip(path):
    with zipfile.ZipFile(path) as z:
        if z.namelist() != ['submission.json']:
            raise ValueError(f'bad zip entries: {path}')
        with z.open('submission.json') as f:
            return json.load(f)


def indexed(rows):
    result = {row['id']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('duplicate query id')
    return result


def digest(value):
    # Canonical serialization hashes UTF-8 chunk_text bytes as well as order and all fields.
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(',', ':')).encode('utf-8')).hexdigest()


def use_chunk_source(reference, source, allow_changed_chunk_docs=False):
    """Keep reference doc order; optionally protect a new subset of those docs."""
    if reference['id'] != source['id']:
        raise ValueError('different ids')
    original = {key(c['doc_id']) for c in reference['relevant_chunks']}
    replacement = {key(c['doc_id']) for c in source['relevant_chunks']}
    if not replacement <= {key(d) for d in reference['relevant_docs']}:
        raise ValueError(f'chunk outside reference docs, id={reference["id"]}')
    if original != replacement and not allow_changed_chunk_docs:
        raise ValueError(f'chunk document set differs from vi37, id={reference["id"]}')
    return {'id': reference['id'], 'relevant_docs': reference['relevant_docs'],
            'relevant_chunks': source['relevant_chunks']}


def build(args):
    rows = load_zip(args.vi41b)
    candidate = {row['id']: {'id': row['id'], 'relevant_docs': row['relevant_docs']} for row in rows}
    if len(candidate) != len(rows):
        raise ValueError('duplicate vi41b ids')
    del rows
    gc.collect()
    rows = load_zip(args.vi37)
    ref = indexed(rows)
    if args.chunks:
        # Retain only document lists and protected doc IDs from vi37 while loading chunks.
        ref = {qid: {'id': qid, 'relevant_docs': row['relevant_docs'],
                     'relevant_chunks': [{'doc_id': d} for d in
                                         {key(c['doc_id']) for c in row['relevant_chunks']}]}
               for qid, row in ref.items()}
        del rows
        gc.collect()
        rows = load_zip(args.chunks)
        source = indexed(rows)
        if set(source) != set(ref):
            raise ValueError('chunk source id set mismatch')
        rows = [use_chunk_source(ref[qid], row, args.allow_changed_chunk_docs) for qid, row in source.items()]
    expected = {json.loads(line)['id'] for line in args.queries.read_text(encoding='utf-8').splitlines() if line.strip()}
    if len(expected) != 1200 or set(ref) != expected or set(candidate) != expected:
        raise ValueError('id set mismatch')
    out, sum_m, affected, insufficient = [], 0, 0, 0
    j37 = j41 = entering = leaving = mismatch = 0
    for row in rows:
        qid = row['id']
        merged, m, short = merge_row(row, candidate[qid])
        a = {key(d) for d in merged['relevant_docs']}
        b = {key(d) for d in row['relevant_docs']}
        c = {key(d) for d in candidate[qid]['relevant_docs']}
        sum_m += m
        affected += m > 0
        insufficient += short > 0
        mismatch += len(a) != len(b)
        j37 += len(a & b) / len(a | b) if a | b else 1
        j41 += len(a & c) / len(a | c) if a | c else 1
        entering += len(a - b)
        leaving += len(b - a)
        out.append(merged)
    if args.chunks and (insufficient or mismatch):
        raise ValueError('insufficient removable docs or doc count mismatch')
    stats = {'queries': len(out), 'M_total': sum_m, 'M_mean': sum_m / len(out),
             'M_positive_queries': affected, 'insufficient_queries': insufficient,
             'doc_count_mismatch': mismatch, 'jaccard_vi37_macro': j37 / len(out),
             'jaccard_vi41b_macro': j41 / len(out), 'doc_in_total_vs_vi37': entering,
             'doc_out_total_vs_vi37': leaving, 'doc_in_mean_vs_vi37': entering / len(out),
             'doc_out_mean_vs_vi37': leaving / len(out)}
    with args.json.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(out, f, ensure_ascii=False)
    # Create the initial one-entry LZMA archive, reordered separately by reorder_zip.py.
    with zipfile.ZipFile(args.initial_zip, 'x', compression=zipfile.ZIP_LZMA) as z:
        with args.json.open('rb') as src, z.open('submission.json', 'w', force_zip64=True) as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)
    args.stats.write_text(json.dumps(stats, indent=2), encoding='utf-8')
    print(json.dumps(stats, indent=2), flush=True)


def audit(args):
    with zipfile.ZipFile(args.final_zip) as z:
        if z.namelist() != ['submission.json'] or z.infolist()[0].compress_type != zipfile.ZIP_LZMA:
            raise ValueError('wrong entries or compression')
        if z.testzip() is not None:
            raise ValueError('CRC error')
        with z.open('submission.json') as f:
            rows = json.load(f)
    actual = indexed(rows)
    expected = {json.loads(line)['id'] for line in args.queries.read_text(encoding='utf-8').splitlines() if line.strip()}
    if len(expected) != 1200 or set(actual) != expected:
        raise ValueError('final ids mismatch')
    final_hashes = {qid: digest(row) for qid, row in actual.items()}
    chunk_hashes = {qid: digest(row['relevant_chunks']) for qid, row in actual.items()}
    lengths = {qid: len(row['relevant_docs']) for qid, row in actual.items()}
    for row in rows:
        docs = [key(d) for d in row['relevant_docs']]
        if len(docs) != len(set(docs)) or not {key(c['doc_id']) for c in row['relevant_chunks']} <= set(docs):
            raise ValueError(f'invalid docs/chunks id={row["id"]}')
    del rows, actual
    gc.collect()
    with args.json.open(encoding='utf-8') as f:
        before = indexed(json.load(f))
    if set(before) != expected or any(digest(row) != final_hashes[qid] for qid, row in before.items()):
        raise ValueError('final content differs from JSON before reorder')
    del before
    gc.collect()
    ref = indexed(load_zip(args.vi37))
    if set(ref) != expected:
        raise ValueError('vi37 ids mismatch')
    count_equal = sum(len(row['relevant_docs']) == lengths[qid] for qid, row in ref.items())
    if args.chunks and count_equal != 1200:
        raise ValueError('doc counts differ from vi37')
    if args.chunks:
        del ref
        gc.collect()
        ref = indexed(load_zip(args.chunks))
        if set(ref) != expected:
            raise ValueError('chunk source ids mismatch')
    chunks_equal = sum(digest(row['relevant_chunks']) == chunk_hashes[qid] for qid, row in ref.items())
    if chunks_equal != 1200:
        raise ValueError('chunks differ from chunk source')
    if args.final_zip.stat().st_size > 99_000_000:
        raise ValueError('zip exceeds size limit')
    print(json.dumps({'crc': 'OK', 'entry': 'submission.json', 'compression': 'ZIP_LZMA',
                      'ids': len(expected), 'content_equal_before_reorder': 1200,
                      ('chunks_equal_source' if args.chunks else 'chunks_equal_vi37'): chunks_equal,
                      'doc_counts_equal_vi37': count_equal,
                      'duplicate_docs': 0, 'chunks_outside_docs': 0}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    b = subs.add_parser('build')
    for name in ('vi37', 'vi41b', 'queries', 'json', 'initial-zip', 'stats'):
        b.add_argument('--' + name, type=Path, required=True)
    b.add_argument('--chunks', type=Path, help='alternate chunks, with same protected docs as vi37')
    b.add_argument('--allow-changed-chunk-docs', action='store_true',
                   help='protect alternate chunk docs within the unchanged vi37 doclist (vi47)')
    a = subs.add_parser('audit')
    for name in ('vi37', 'queries', 'json', 'final-zip'):
        a.add_argument('--' + name, type=Path, required=True)
    a.add_argument('--chunks', type=Path, help='alternate chunk source for audit')
    args = parser.parse_args()
    (build if args.command == 'build' else audit)(args)


if __name__ == '__main__':
    main()
