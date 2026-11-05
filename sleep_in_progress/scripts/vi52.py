"""vi52: vi41c sources + d8w + q3e, fixed RRF; one local build only."""
import argparse
import gc
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from fuse3 import fuse3
from vi37 import docwin_hits
from vi50 import protect
from vi48 import fingerprint, read, write

TEMP = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))


def fused(data, bge, d8w=None, q3e=None):
    extras = [(docwin_hits(data['pool'], bge), 1.)]
    for scores in (d8w, q3e):
        if scores is not None:
            extras.append((docwin_hits(data['pool'], scores), 1.))
    return fuse3(data['pool'], data['q8b'], data['dense'], (1., 2., 2., 1.),
                 60, data['hyde'], extras)


def doclists(base, rankings):
    return {r['id']: protect(r, rankings[str(r['id'])])[0]['relevant_docs'] for r in base}


def compare(actual, reference):
    assert set(actual) == set(reference)
    n = len(actual)
    entries = sum(len(set(actual[q]) - set(reference[q])) for q in actual)
    return dict(queries=n, exact=sum(actual[q] == reference[q] for q in actual),
                changed_sets=sum(set(actual[q]) != set(reference[q]) for q in actual),
                docs_entering=entries, docs_entering_per_query=entries / n if n else 0.)


def load_docs(path):
    from vi41c import indexed, load_zip
    rows = indexed(load_zip(path))
    assert set(rows) == set(range(1, 1201))
    return {q: r['relevant_docs'] for q, r in rows.items()}


def build(out, cosine=TEMP/'vi51/cosines_vi51a.json',
          reference51_path=TEMP/'vi51a/submission_vi51a.zip', name='vi52',
          reference52_path=None, sanity_cosine=None):
    import dwmax_diag as dw
    from vi41c import digest, indexed, load_zip
    from vi29 import assert_clean, load_corpus_ids
    from reorder_zip import write_reordered_zip

    start = time.monotonic()
    out.mkdir(parents=True, exist_ok=True)
    if (reference52_path is None) != (sanity_cosine is None):
        raise ValueError('reference52 and sanity cosine must be supplied together')
    zp, jp, intent = out/f'submission_{name}.zip', out/f'submission_{name}.json', out/'build_intent.json'
    assert not zp.exists() and not jp.exists() and not intent.exists() and not (out/'build_report.json').exists(), 'one build only'
    paths = [TEMP/'vi50/fusion_context.json.zst', TEMP/'vi50/d8w.json',
             cosine, TEMP/'vi48/submission_vi48.zip',
             TEMP/'vi50/submission_vi50.zip', reference51_path]
    if reference52_path is not None:
        paths.extend([reference52_path, sanity_cosine])
    before = {str(p): fingerprint(p) for p in paths}
    ctx = dw.read(paths[0]); data, bge = ctx['data'], ctx['bge']
    d8w, q3e = read(paths[1]), read(paths[2])
    assert set(q3e) == set(data['pool']) == set(d8w)
    for q, hits in data['pool'].items():
        assert set(q3e[q]) == {str(h[0]) for h in hits}
        assert set(d8w[q]) <= {str(h[0]) for h in hits}
        assert all(math.isfinite(v) for v in q3e[q].values())
        assert all(math.isfinite(v) for v in d8w[q].values())
    base = sorted(load_zip(paths[3]), key=lambda r: r['id'])
    assert len(base) == 1200 and set(indexed(base)) == set(range(1, 1201))
    reference50 = load_docs(paths[4]); gc.collect()
    reference51 = load_docs(paths[5]); gc.collect()
    sanity50 = compare(doclists(base, fused(data, bge, d8w=d8w)), reference50)
    sanity51 = compare(doclists(base, fused(data, bge, q3e=q3e)), reference51)
    reference51_label = 'vi51a' if name == 'vi52' else 'vi51b'
    sanity = {'without_q3e_vs_vi50': sanity50, f'without_d8w_vs_{reference51_label}': sanity51}
    if reference52_path is not None:
        reference52 = load_docs(reference52_path); gc.collect()
        q3e_sanity = read(sanity_cosine)
        assert set(q3e_sanity) == set(data['pool'])
        for q, hits in data['pool'].items():
            assert set(q3e_sanity[q]) == {str(h[0]) for h in hits}
            assert all(math.isfinite(v) for v in q3e_sanity[q].values())
        sanity['cosine_06b_vs_vi52'] = compare(doclists(base, fused(data, bge, d8w, q3e_sanity)), reference52)
        del q3e_sanity
    write(out/'sanity.json', sanity)
    if any(result['exact'] != 1200 for result in sanity.values()):
        raise RuntimeError(f'sanity mismatch; stopped before zip: {sanity}')
    rankings = fused(data, bge, d8w, q3e)
    rows = []; protected = 0
    for r in base:
        row, missing = protect(r, rankings[str(r['id'])])
        protected += missing
        rows.append(row)
    lists = {r['id']: r['relevant_docs'] for r in rows}
    offline = {'vs_vi50': compare(lists, reference50), f'vs_{reference51_label}': compare(lists, reference51),
               'protected_docs': protected}
    if reference52_path is not None:
        offline['vs_vi52'] = compare(lists, reference52)
    write(out/'offline.json', offline)
    hashes_chunks = {r['id']: digest(r['relevant_chunks']) for r in base}
    counts = {r['id']: len(r['relevant_docs']) for r in base}
    source = TEMP/'vi45'
    bad = set(map(int, read(source/'bad_ids.json'))) - set(map(int, read(source/'new_ids.json')))
    clean = assert_clean(rows, bad, load_corpus_ids(source/'links_corpus_vi_en_zh.parquet'))
    assert all(len(r['relevant_docs']) == len(set(r['relevant_docs'])) for r in rows)
    assert all(digest(r['relevant_chunks']) == hashes_chunks[r['id']] and len(r['relevant_docs']) == counts[r['id']] for r in rows)
    hashes = {r['id']: digest(r) for r in rows}
    del ctx, data, bge, d8w, q3e, base, rankings, reference50, reference51, lists
    gc.collect()
    # Durable exclusive guard: if the compression fails, do not repeat it.
    with intent.open('x', encoding='utf-8') as f:
        f.write('{"build_attempts":1}')
    info = write_reordered_zip(rows, zp)
    assert zp.stat().st_size <= 99_000_000
    del rows; gc.collect()
    try:
        with zipfile.ZipFile(zp) as z:
            assert z.namelist() == ['submission.json'] and z.infolist()[0].compress_type == zipfile.ZIP_LZMA
            assert z.testzip() is None
            with z.open('submission.json') as src, jp.open('xb') as dst:
                shutil.copyfileobj(src, dst)
        final = load_zip(zp)
        assert {r['id']: digest(r) for r in final} == hashes
        assert all(digest(r['relevant_chunks']) == hashes_chunks[r['id']] and len(r['relevant_docs']) == counts[r['id']] for r in final)
        del final; gc.collect()
        subprocess.run([sys.executable, str(Path(__file__).with_name('evaluate.py')), 'validate', str(jp)], check=True)
        subprocess.run([sys.executable, str(Path(__file__).resolve()), 'check-zip', '--out', str(out), '--name', name], check=True)
        assert all(fingerprint(p) == meta for p, meta in ((Path(k), v) for k, v in before.items()))
        report = dict(**info, **fingerprint(zp), validate=0, check_zip=0, assert_clean=clean,
                      crc=0, ids=1200, chunks_equal_vi48=1200, doc_counts_equal_vi48=1200,
                      no_duplicate_docs=True, entry='submission.json', compression='LZMA',
                      build_attempts=1, inputs_unchanged=True, seconds=time.monotonic()-start)
        write(out/'build_report.json', report)
        print('BUILD_DONE', report, flush=True)
    finally:
        if jp.exists():
            jp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['build', 'check-zip'])
    ap.add_argument('--out', type=Path, default=TEMP/'vi52')
    ap.add_argument('--cosine', type=Path, default=TEMP/'vi51/cosines_vi51a.json')
    ap.add_argument('--reference51', type=Path, default=TEMP/'vi51a/submission_vi51a.zip')
    ap.add_argument('--reference52', type=Path)
    ap.add_argument('--sanity-cosine', type=Path)
    ap.add_argument('--name', choices=['vi52', 'vi53'], default='vi52')
    args = ap.parse_args()
    if args.stage == 'check-zip':
        from vi23 import check_zip
        raise SystemExit(check_zip(str(args.out/f'submission_{args.name}.json'), str(args.out/f'submission_{args.name}.zip'),
                                   str(args.out/'check_zip.json'), args.name, max_zip=99_000_000))
    build(args.out, args.cosine, args.reference51, args.name, args.reference52, args.sanity_cosine)


if __name__ == '__main__':
    main()
