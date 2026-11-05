"""CPU-only vi42 bridges on every vi37 chunk-bearing document; no tuning."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import vi42 as v
from vi41c import indexed, load_zip

ROOT = v.ROOT
OUT = ROOT / 'data/sub/vi44'


def select_bridges(item, docs, windows, geometry=None):
    selected, seen = {}, set()
    for c in item['relevant_chunks']:
        d = str(c['doc_id'])
        selected.setdefault(d, set()).add(c['chunk_text'])
        seen.add((d, c['chunk_text']))
    geometry = geometry if geometry is not None else {}
    out = []
    for rank, did in enumerate(item['relevant_docs'], 1):
        d = str(did)
        if d not in selected:
            continue
        if d not in geometry:
            geometry[d] = v.bridge_geometry(docs[int(d)])
        chunks, bridges = geometry[d]
        assert selected[d] <= {c.text for c in chunks}, ('selected U900 grid mismatch', item['id'], d)
        if not bridges:
            continue
        assert d in windows, ('missing cached windows', item['id'], d)
        for b in bridges:
            i = b['index']
            if chunks[i].text not in selected[d] and chunks[i+1].text not in selected[d]:
                continue
            sc = v.gate_score(windows[d], b['band'])
            if sc is None or (d, b['text']) in seen:
                continue
            seen.add((d, b['text']))
            out.append({'doc_id': did, 'chunk_text': b['text'], 'score': sc,
                        'rank': rank, 'boundary_index': i, 'start': b['start'], 'end': b['end']})
    return sorted(out, key=lambda b: (-b['score'], b['rank'], b['boundary_index']))


def rank_buckets(bridges, threshold=None):
    rows = [b for bs in bridges.values() for b in bs if threshold is None or b['score'] >= threshold]
    return {'1-50': sum(b['rank'] <= 50 for b in rows),
            '51-100': sum(51 <= b['rank'] <= 100 for b in rows),
            '101+': sum(b['rank'] >= 101 for b in rows)}


def ordered_subsequence(small, large):
    it = iter(large)
    return all(any(x == y for y in it) for x in small)


def check_relations(sub, ref37, ref42, bridges, threshold):
    a, b, c = indexed(sub), indexed(ref37), indexed(ref42)
    assert set(a) == set(b) == set(c) == set(range(1, 1201))
    for q, row in a.items():
        old = b[q]['relevant_chunks']
        chunks = row['relevant_chunks']
        assert row['relevant_docs'] == b[q]['relevant_docs']
        assert chunks[:len(old)] == old
        expected = v.append_bridges([b[q]], {str(q): bridges[str(q)]}, threshold,
                                    enabled=threshold is not None)[0]['relevant_chunks']
        assert chunks == expected
        # vi37 is a literal prefix. vi42's appended bridges retain their relative
        # order, but new high-scoring bridges may precede them: ordered subsequence.
        assert c[q]['relevant_chunks'][:len(old)] == old
        assert ordered_subsequence(c[q]['relevant_chunks'][len(old):], chunks[len(old):])
        assert {x['doc_id'] for x in chunks} == {x['doc_id'] for x in old}
    return {'doclists_equal_vi37': 1200, 'vi37_literal_prefix': 1200,
            'vi42_ordered_subsequence': 1200, 'bridge_order_and_append': 1200,
            'chunk_bearing_doc_sets_unchanged': 1200}


def final_checks(sub, zp, name, out):
    ordered = out / f'submission_{name}.json'
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ['submission.json']
        assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        assert z.testzip() is None
        with z.open('submission.json') as src, ordered.open('wb') as dst:
            shutil.copyfileobj(src, dst)
    subprocess.run([sys.executable, str(Path(__file__).parent/'evaluate.py'), 'validate', str(ordered)], check=True)
    code = v.vi23.check_zip(str(ordered), str(zp), str(out/f'report_chain_{name}.json'), name, max_zip=v.LIMIT)
    assert code == 0
    assert v.compare_by_id(v.dw.read(ordered), sub) == {'count': 1200, 'docs': 1200, 'chunks': 1200}
    bad = set(map(int, v.dw.read(ROOT/'data/sub/diag28/bad_ids.json'))) - set(map(int, v.dw.read(ROOT/'data/sub/diag30/new_ids.json')))
    corpus = v.vi29.load_corpus_ids(ROOT/'data/sub/diag19/links_corpus_vi_en_zh.parquet')
    clean = v.vi29.assert_clean(sub, bad, corpus)
    assert {x['id'] for x in sub} == set(range(1, 1201))
    return {'validate': 0, 'check_zip': code, 'crc': 0, 'assert_clean': clean,
            'entry': 'submission.json', 'compression': 'LZMA', 'ids': 1200}


def build():
    start = time.monotonic()
    inputs = [v.vi37.OUT/'submission_vi37.json', v.vi37.OUT/'docwin_new.json.zst',
              v.vi37.OUT/'cache_fhyde_v37.json', v.vi37.DIAG/'docwin_max_provenance.json',
              ROOT/'data/sub/vi42/submission_vi42.zip', ROOT/'data/sub/vi42/bridges_vi42.json.zst']
    before = {str(p): v.fingerprint(p) for p in inputs}
    ref = v.dw.read(inputs[0])
    ref42 = load_zip(inputs[4])
    needed = {str(it['id']): {str(c['doc_id']) for c in it['relevant_chunks']} for it in ref}
    from rerank_run import load_docs
    wanted = {int(d) for ds in needed.values() for d in ds}
    files = sorted((v.vi37.OUT/'verify_text').glob('docs_verify.part*.jsonl.zst'))
    assert files
    docs = load_docs([str(p) for p in files], wanted=wanted)
    assert wanted <= set(docs)
    wins, used = v.load_windows(needed)
    adjusted = []
    for q, ds in wins.items():
        for d, ws in ds.items():
            grid = v.window_grid(docs[int(d)])
            if [(s, e) for s, e, lg in ws] != grid:
                assert len(ws) == len(grid), ('window count mismatch', q, d, len(ws), len(grid))
                ds[d] = [[s, e, w[2]] for (s, e), w in zip(grid, ws)]
                adjusted.append([q, d])
    geometry = {}
    bridges = {str(it['id']): select_bridges(it, docs, wins[str(it['id'])], geometry) for it in ref}
    v.vi37.writez(OUT/'bridges_vi44.json.zst', bridges)
    top50 = {q: [b for b in bs if b['rank'] <= 50] for q, bs in bridges.items()}
    assert top50 == v.dw.read(inputs[5]), 'top50 bridge manifest differs from vi42'
    rep42 = v.dw.read(ROOT/'data/sub/vi42/build_report_vi42.json')
    sanity = v.compare_by_id(v.append_bridges(ref, top50, rep42['threshold'], enabled=rep42['threshold'] is not None), ref42)
    assert sanity == {'count': 1200, 'docs': 1200, 'chunks': 1200}
    pre = {**v.offline(ref, bridges), 'doc_rank_buckets': rank_buckets(bridges)}
    v.dw.write(OUT/'selection_report.json', {'sanity_top50_vi42': sanity, 'before_budget': pre,
               'historical_layers': used, 'ordinal_offset_remap_pairs': adjusted})
    print('SELECTED', json.dumps(pre), flush=True)
    del wins, geometry
    attempts, zp = [], OUT/'submission_vi44.zip'
    def measure(thr):
        result = v.write_reordered_zip(v.append_bridges(ref, bridges, thr, enabled=thr is not None), zp)
        attempts.append({'threshold': thr, **result})
        v.dw.write(OUT/'budget_attempts.json', attempts)
        print('BUDGET', json.dumps(attempts[-1]), flush=True)
        return result['zip_bytes']
    threshold, _ = v.budget_threshold([b['score'] for bs in bridges.values() for b in bs], measure)
    sub = v.append_bridges(ref, bridges, threshold, enabled=threshold is not None)
    # The last measured probe need not be the chosen threshold.
    info = v.write_reordered_zip(sub, zp)
    assert info['zip_bytes'] <= v.LIMIT
    relations = check_relations(sub, ref, ref42, bridges, threshold)
    checks = final_checks(sub, zp, 'vi44', OUT)
    verbatim = sum(c['chunk_text'] not in docs[int(c['doc_id'])] for it in sub for c in it['relevant_chunks'])
    assert verbatim == 0
    assert before == {str(p): v.fingerprint(p) for p in inputs}
    post = {**v.offline(ref, bridges, threshold), 'doc_rank_buckets': rank_buckets(bridges, threshold)} if threshold is not None else v.offline(ref, {})
    report = {**info, **v.fingerprint(zp), **checks, **relations, 'verbatim_bad': verbatim,
              'sanity_top50_vi42': sanity, 'threshold': threshold, 'attempts': attempts,
              'before_budget': pre, 'after_budget': post, 'ordinal_offset_remap_pairs': adjusted,
              'source_fingerprints': before, 'seconds': time.monotonic()-start,
              'vi42_preservation_definition': 'vi37 literal prefix; vi42 bridge suffix ordered subsequence of vi44 bridge suffix',
              'budget_note': 'vi42 global whole-score-group rule; actual reordered ZIP measured'}
    v.dw.write(OUT/'build_report_vi44.json', report)
    print('DONE', json.dumps(report), flush=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with v.wait_lock(ROOT/'locks/heavy_job.lock', f'vi44 pid={os.getpid()}'):
        build()


if __name__ == '__main__':
    main()
