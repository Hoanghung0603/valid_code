"""Independent checker for CX-chain candidates. Stdlib only.

Keeps RAM low: base is read first and reduced to per-query docs and per-chunk hashes,
then released before the candidate is loaded.
"""
import argparse
import gc
import hashlib
import json
import math
import zipfile
from pathlib import Path

LIMIT = 99_000_000
IDS = set(range(1, 1201))


def canon(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))


def chash(value):
    return hashlib.sha256(canon(value).encode('utf-8')).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with zipfile.ZipFile(path) as z:
        if z.namelist() != ['submission.json']:
            raise ValueError(f'bad zip entries: {path}')
        with z.open('submission.json') as f:
            return json.load(f)


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0
    k = max(0, math.ceil(p / 100 * len(sorted_vals)) - 1)
    return sorted_vals[k]


def load_base_summary(path):
    rows = read_rows(path)
    summary = {}
    for row in rows:
        qid = int(row['id'])
        if qid in summary:
            raise ValueError(f'duplicate base id {qid}')
        chunks = row['relevant_chunks']
        summary[qid] = {
            'docs': [str(d) for d in row['relevant_docs']],
            'chunk_hashes': [chash(c) for c in chunks],
        }
    del rows
    gc.collect()
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--cand', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    cand_bytes = args.cand.stat().st_size
    with zipfile.ZipFile(args.cand) as z:
        names_ok = z.namelist() == ['submission.json']
        comp_ok = bool(z.infolist()) and z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        crc_ok = z.testzip() is None
    size_ok = cand_bytes <= LIMIT

    base = load_base_summary(args.base)
    base_ids_ok = set(base) == IDS
    gc.collect()

    rows = read_rows(args.cand)
    cand_ids = [int(r['id']) for r in rows]
    ids_ok = len(cand_ids) == len(set(cand_ids)) and set(cand_ids) == IDS

    docs_equal = prefix_equal = dup_pair_queries = outside_queries = 0
    appended = []
    for row in rows:
        qid = int(row['id'])
        b = base.get(qid)
        if b is None:
            continue
        docs = [str(d) for d in row['relevant_docs']]
        if docs == b['docs']:
            docs_equal += 1
        chunks = row['relevant_chunks']
        n = len(b['chunk_hashes'])
        hashes = [chash(c) for c in chunks]
        if len(chunks) >= n and hashes[:n] == b['chunk_hashes']:
            prefix_equal += 1
        pairs = [(str(c['doc_id']), c['chunk_text']) for c in chunks]
        if len(pairs) != len(set(pairs)):
            dup_pair_queries += 1
        docset = set(docs)
        if any(str(c['doc_id']) not in docset for c in chunks):
            outside_queries += 1
        appended.append(max(0, len(chunks) - n))

    appended_sorted = sorted(appended)
    total_appended = sum(appended)
    stats = {
        'bytes': cand_bytes,
        'sha256': file_sha256(args.cand),
        'ids': len(cand_ids),
        'ids_ok': ids_ok,
        'base_ids_ok': base_ids_ok,
        'docs_equal': docs_equal,
        'prefix_equal': prefix_equal,
        'queries_with_duplicate_chunk_pairs': dup_pair_queries,
        'queries_with_chunk_outside_docs': outside_queries,
        'total_appended_chunks': total_appended,
        'appended_per_query': {
            'mean': total_appended / len(appended) if appended else 0,
            'p50': pct(appended_sorted, 50),
            'p90': pct(appended_sorted, 90),
            'max': appended_sorted[-1] if appended_sorted else 0,
        },
        'queries_with_no_appended_chunk': sum(1 for a in appended if a == 0),
        'checks': {
            'size_le_99000000': size_ok,
            'entries_submission_json_only': names_ok,
            'compress_ZIP_LZMA': comp_ok,
            'testzip_ok': crc_ok,
        },
    }
    passed = (size_ok and names_ok and comp_ok and crc_ok and ids_ok and base_ids_ok
              and docs_equal == 1200 and prefix_equal == 1200
              and dup_pair_queries == 0 and outside_queries == 0)
    stats['pass'] = passed

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(stats, indent=2, ensure_ascii=False), flush=True)
    raise SystemExit(0 if passed else 1)


if __name__ == '__main__':
    main()
