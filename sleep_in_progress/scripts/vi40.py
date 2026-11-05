"""CPU-only vi37 -> immutable dchunk 512/50 offsets. Never submit or call GPU."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import zipfile

ROOT = Path(os.environ.get('R2AI_ROOT', 'D:/r2ai'))
OUT = ROOT / 'data/sub/vi40'


def intersection(a, b, c, d):
    return max(0, min(b, d) - max(a, c))


def ranked_windows(start, end, windows):
    """Positive overlap only; deterministic overlap-desc/chunk_id-asc."""
    return sorted((w for w in windows if intersection(start, end, w['start'], w['end'])),
                  key=lambda w: (-intersection(start, end, w['start'], w['end']), w['chunk_id']))


def union_length(spans):
    end, total = -1, 0
    for a, b in sorted(spans):
        total += max(0, b-max(a, end))
        end = max(end, b)
    return total


def locate(body, full, chunk):
    """Translate body-only source coordinates to title+body export; never decode tokens."""
    stripped = body.strip()
    lead = len(body)-len(body.lstrip())
    shift = full.find(stripped)
    assert stripped and shift >= 0, 'Body differs from immutable dchunk export'
    pos = body.find(chunk)
    assert pos >= lead, 'Source chunk absent from verified body'
    start = shift+pos-lead
    end = start+len(chunk)
    assert full[start:end] == chunk
    return start, end, shift-lead


def remap(items, mappings, fulltexts, k):
    if k not in (1, 2):
        raise ValueError('Only predeclared k=1 or k=2')
    out, selected = [], []
    for item in items:
        seen, chunks, provenance = set(), [], []
        for source in item['relevant_chunks']:
            d = int(source['doc_id'])
            for w in mappings[(d, source['chunk_text'])][:k]:
                text = fulltexts[d][w['start']:w['end']]
                assert text
                key = (d, text)
                if key in seen:
                    continue
                seen.add(key)
                chunks.append(dict(doc_id=source['doc_id'], chunk_text=text))
                provenance.append(w)
        assert {int(c['doc_id']) for c in chunks} == {int(c['doc_id']) for c in item['relevant_chunks']}
        out.append(dict(id=item['id'], relevant_docs=item['relevant_docs'][:], relevant_chunks=chunks))
        selected.append(provenance)
    return out, selected


def describe(values):
    import numpy as np
    a = np.asarray(values, dtype=float)
    assert a.size and np.isfinite(a).all()
    return dict(n=len(a), mean=float(a.mean()), min=float(a.min()), max=float(a.max()),
                **{f'p{p}': float(np.percentile(a, p)) for p in (5, 50, 95)})


def ws_spans(text):
    spans = [m.span() for m in re.finditer(r'\S+', text)]
    out = []
    for i in range(0, len(spans), 250):
        j = min(i+300, len(spans))
        out.append((spans[i][0], spans[j-1][1]))
        if j == len(spans):
            break
    return out


def run():
    import pyarrow.parquet as pq
    import vi29
    from dchunk import sha, write
    from dwmax_diag import read
    from rerank_run import load_docs
    from reorder_zip import write_reordered_zip
    from vi23 import check_zip
    t0 = time.monotonic()
    source = ROOT/'data/sub/vi37/submission_vi37.json'
    before_sha = sha(source)
    items = read(source)
    assert len(items) == 1200 and {it['id'] for it in items} == set(range(1, 1201))
    wanted = {int(c['doc_id']) for it in items for c in it['relevant_chunks']}
    dc = ROOT/'data/sub/dchunk'
    manifest = read(dc/'export_report.json')
    assert (manifest['content_tokens'], manifest['overlap'], manifest['stride']) == (512, 50, 462)
    fulltexts, windows = {}, {}
    for part in manifest['files']:
        for row in pq.read_table(dc/part['documents'], filters=[('doc_id', 'in', sorted(wanted))]).to_pylist():
            assert row['doc_id'] not in fulltexts
            fulltexts[row['doc_id']] = row['text']
        for row in pq.read_table(dc/part['metadata'], filters=[('doc_id', 'in', sorted(wanted))]).to_pylist():
            d = row['doc_id']
            assert 0 <= row['start'] < row['end'] <= len(fulltexts[d])
            assert 0 < row['tokens'] <= 512
            windows.setdefault(d, []).append(row)
    assert set(fulltexts) == set(windows) == wanted
    bodypaths = sorted((ROOT/'data/sub/vi37/verify_text').glob('docs_verify.part*.jsonl.zst'))
    assert bodypaths, 'Need vi37 verified body exports; do not fall back to stale/raw offsets'
    bodies = load_docs([str(p) for p in bodypaths], wanted=wanted)
    assert set(bodies) == wanted
    mappings, source_spans, shifts = {}, {}, {}
    for it in items:
        for c in it['relevant_chunks']:
            d, text = int(c['doc_id']), c['chunk_text']
            key = (d, text)
            if key not in mappings:
                a, b, shift = locate(bodies[d], fulltexts[d], text)
                source_spans[key], shifts[d] = (a, b), shift
                mappings[key] = ranked_windows(a, b, windows[d])
                assert mappings[key], ('No intersecting grid window', d)
    overlaps, top1u, top1w, top2u, allu, allratio = [], [], [], [], [], []
    pair_counts, grid_docwin_counts = [], []
    for it in items:
        for d in {int(c['doc_id']) for c in it['relevant_chunks']}:
            pair_counts.append(len(windows[d]))
        for c in it['relevant_chunks']:
            key = (int(c['doc_id']), c['chunk_text'])
            a, b = source_spans[key]
            ws = mappings[key]
            ov = [intersection(a, b, w['start'], w['end']) for w in ws]
            overlaps.append(len(ws))
            allratio.extend(v/(b-a) for v in ov)
            top1u.append(ov[0]/(b-a))
            top1w.append(ov[0]/(ws[0]['end']-ws[0]['start']))
            def coverage(rows):
                return union_length([(max(a,w['start']),min(b,w['end'])) for w in rows])/(b-a)
            top2u.append(coverage(ws[:2]))
            allu.append(coverage(ws))
    for d in wanted:
        spans = [(a+shifts[d], b+shifts[d]) for a,b in ws_spans(bodies[d])]
        grid_docwin_counts.extend(sum(intersection(a,b,w['start'],w['end']) > 0 for a,b in spans)
                                  for w in windows[d])
    survey = dict(source_sha256=before_sha, dchunk_manifest_sha256=sha(dc/'export_report.json'),
                  queries=1200, source_chunks=sum(len(it['relevant_chunks']) for it in items),
                  unique_source_chunks=len(mappings), unique_chunk_docs=len(wanted),
                  query_doc_pairs=len(pair_counts), dchunk_windows_per_query_doc=describe(pair_counts),
                  dchunk_windows_per_unique_doc=describe([len(windows[d]) for d in wanted]),
                  intersecting_windows_per_U900=describe(overlaps),
                  each_intersection_over_U900=describe(allratio),
                  top1_intersection_over_U900=describe(top1u), top1_intersection_over_window=describe(top1w),
                  top2_union_over_U900=describe(top2u), all_union_over_U900=describe(allu),
                  geometric_docwin_intersections_per_dchunk=describe(grid_docwin_counts),
                  docwin_note='300 whitespace words/stride250, XLM-R pair cap512 only_second; char spans pre-truncation. 8B is doc-level, not these windows.',
                  acceptance='Final > 0.3419 AND Chunks F2 > 0.3142; no scores queried', gpu_seconds=0, usd=0)
    write(OUT/'survey.json', survey)
    print('SURVEY DONE', len(mappings), len(wanted), flush=True)
    bad = set(map(int, read(ROOT/'data/sub/diag28/bad_ids.json'))) - set(map(int, read(ROOT/'data/sub/diag30/new_ids.json')))
    corpus = vi29.load_corpus_ids(ROOT/'data/sub/diag19/links_corpus_vi_en_zh.parquet')
    reports = {}
    for k, name in [(1,'vi40'), (2,'vi40_top2')]:
        sub, provenance = remap(items, mappings, fulltexts, k)
        clean = vi29.assert_clean(sub, bad, corpus)
        expected = {it['id']:it['relevant_docs'] for it in items}
        assert {it['id']:it['relevant_docs'] for it in sub} == expected
        verbatim_bad = sum(fulltexts[int(c['doc_id'])][w['start']:w['end']] != c['chunk_text']
                           for it, ws in zip(sub, provenance) for c,w in zip(it['relevant_chunks'],ws))
        assert verbatim_bad == 0
        zp, jp = OUT/f'submission_{name}.zip', OUT/f'submission_{name}.json'
        packaging = write_reordered_zip(sub, zp)
        with zipfile.ZipFile(zp) as z:
            assert z.namelist() == ['submission.json'] and z.infolist()[0].compress_type == zipfile.ZIP_LZMA
            with z.open('submission.json') as src, jp.open('wb') as dst:
                shutil.copyfileobj(src, dst, 8<<20)
        subprocess.run([sys.executable, str(ROOT/'scripts/evaluate.py'), 'validate', str(jp)], check=True)
        zcheck = check_zip(str(jp),str(zp),str(OUT/f'zip_report_{name}.json'),name,max_zip=99_000_000)
        assert zcheck == 0 and zp.stat().st_size <= 99_000_000
        nchunk = sum(len(it['relevant_chunks']) for it in sub)
        rep = dict(k=k, **packaging, sha256=sha(zp), validate_exit=0, check_zip=zcheck,
                   assert_clean=clean, verbatim_bad=verbatim_bad, doclists_identical=1200,
                   chunks=nchunk, chunks_per_query=nchunk/1200,
                   chunk_source_docs_per_query=sum(len({c['doc_id'] for c in it['relevant_chunks']}) for it in sub)/1200,
                   whitespace_tokens_per_chunk=describe([len(c['chunk_text'].split()) for it in sub for c in it['relevant_chunks']]),
                   exported_XLMR_content_tokens_per_chunk=describe([w['tokens'] for ws in provenance for w in ws]),
                   zip_entries=1, zip_compression='LZMA')
        reports[name] = rep
        write(OUT/f'build_report_{name}.json',rep)
        print('BUILD DONE',name,json.dumps(rep),flush=True)
    assert sha(source) == before_sha
    survey['baseline'] = dict(chunks_per_query=survey['source_chunks']/1200,
        chunk_source_docs_per_query=survey['query_doc_pairs']/1200,
        whitespace_tokens_per_chunk=describe([len(c['chunk_text'].split()) for it in items for c in it['relevant_chunks']]))
    survey['builds'] = reports
    survey['elapsed_seconds'] = time.monotonic()-t0
    write(OUT/'survey.json', survey)


def status():
    lock = ROOT/'locks/heavy_job.lock'
    obj = dict(lock=lock.read_text(encoding='utf-8') if lock.exists() else None)
    for name in ('survey.json','build_report_vi40.json','build_report_vi40_top2.json'):
        p = OUT/name
        if p.exists():
            obj[name] = json.loads(p.read_text(encoding='utf-8'))
    log = OUT/'build.log'
    if log.exists():
        with log.open('rb') as f:
            f.seek(max(0,log.stat().st_size-2500))
            obj['log_tail'] = f.read().decode('utf-8','replace')
        obj['log_mtime'] = log.stat().st_mtime
    print(json.dumps(obj,ensure_ascii=False))
    subprocess.run(['schtasks','/Query','/TN','r2ai_vi40','/FO','LIST','/V'],check=False)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['build','status'])
    a = ap.parse_args()
    if a.stage == 'status':
        return status()
    from dwmax_diag import owned_lock
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'build_report_vi40.json').exists(), 'Refuse completed-build overwrite'
    with owned_lock(ROOT/'locks/heavy_job.lock',f'vi40 CPU pid={os.getpid()}'):
        run()


if __name__ == '__main__':
    main()
