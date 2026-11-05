"""CPU-only vi41c doclist plus untouched vi42 chunks; never submits."""
import os
import time

import vi44 as v
from vi41c import indexed, load_zip, merge_row

ROOT = v.ROOT
OUT = ROOT/'data/sub/vi43'
SOURCE_SHA = 'b1b01a9f1f5e93e512273432ca2b3bf75da55951252297a75847806683a85b69'


def combine(ref37, ref42, candidates):
    a, b, c = indexed(ref37), indexed(ref42), indexed(candidates)
    assert set(a) == set(b) == set(c) == set(range(1, 1201))
    out, missing, insufficient = [], 0, 0
    for row in ref42:
        q = row['id']
        assert {str(x['doc_id']) for x in row['relevant_chunks']} == {
            str(x['doc_id']) for x in a[q]['relevant_chunks']}, ('chunk-bearing docs changed', q)
        merged, m, short = merge_row(row, c[q])
        missing += m
        insufficient += short
        assert len(merged['relevant_docs']) == len(a[q]['relevant_docs'])
        assert merged['relevant_chunks'] == row['relevant_chunks']
        # vi41c already protects exactly this set, so its doclist must be unchanged.
        assert merged['relevant_docs'] == c[q]['relevant_docs']
        out.append(merged)
    return out, {'queries': 1200, 'chunks_equal_vi42': 1200, 'docs_equal_vi41c': 1200,
                 'doc_counts_equal_vi37': 1200, 'chunk_bearing_doc_sets_equal_vi37': 1200,
                 'missing_chunk_docs_appended': missing, 'insufficient_doc_removals': insufficient,
                 'docs_per_query': sum(len(x['relevant_docs']) for x in out)/1200,
                 'chunks_per_query': sum(len(x['relevant_chunks']) for x in out)/1200}


def build():
    start = time.monotonic()
    paths = [ROOT/'data/sub/vi37/submission_vi37.json', ROOT/'data/sub/vi42/submission_vi42.zip',
             OUT/'source_vi41c.zip']
    before = {str(p): v.v.fingerprint(p) for p in paths}
    assert before[str(paths[2])]['sha256'] == SOURCE_SHA
    ref37, ref42, candidates = v.v.dw.read(paths[0]), load_zip(paths[1]), load_zip(paths[2])
    sub, stats = combine(ref37, ref42, candidates)
    zp = OUT/'submission_vi43.zip'
    info = v.v.write_reordered_zip(sub, zp)
    assert info['zip_bytes'] <= v.v.LIMIT
    checks = v.final_checks(sub, zp, 'vi43', OUT)
    from rerank_run import load_docs
    wanted = {int(c['doc_id']) for it in sub for c in it['relevant_chunks']}
    files = sorted((ROOT/'data/sub/vi37/verify_text').glob('docs_verify.part*.jsonl.zst'))
    assert files
    docs = load_docs([str(p) for p in files], wanted=wanted)
    assert wanted <= set(docs)
    verbatim = sum(c['chunk_text'] not in docs[int(c['doc_id'])] for it in sub for c in it['relevant_chunks'])
    assert verbatim == 0
    assert before == {str(p): v.v.fingerprint(p) for p in paths}
    report = {**info, **v.v.fingerprint(zp), **checks, **stats, 'verbatim_bad': verbatim,
              'source_fingerprints': before, 'seconds': time.monotonic()-start,
              'rule': 'vi41c docs + untouched vi42 chunks; vi41c protected-doc repair; no repair needed'}
    v.v.dw.write(OUT/'build_report_vi43.json', report)
    print('DONE', v.json.dumps(report), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with v.v.wait_lock(ROOT/'locks/heavy_job.lock', f'vi43 pid={os.getpid()}'):
        build()


if __name__ == '__main__':
    main()
