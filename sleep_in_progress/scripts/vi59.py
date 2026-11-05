"""vi59d: fixed top-200 expansion, immutable vi56 reuse, unchanged vi56/vi52 math.

All paths are explicit. Never writes historical TEMP or replay directories.
"""
import argparse
from collections import defaultdict
import gc
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

import vi56
from vi55 import read, write, iterz, fingerprint


def select(top, pool, allowed, limit=200):
    assert limit in (100, 200) and set(top) == set(pool)
    result = {}
    for q, hits in pool.items():
        old = {int(h[0]) for h in hits}
        first = top[q][:limit]
        assert len(first) == limit == len({int(h[0]) for h in first})
        result[q] = [int(d) for d, _ in first if int(d) not in old and int(d) in allowed]
    return result


def copy_checked(src, dst):
    src, dst = Path(src), Path(dst)
    before = fingerprint(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        assert fingerprint(dst) == before, ('changed owned copy', dst)
    else:
        shutil.copy2(src, dst)
    assert fingerprint(dst) == before == fingerprint(src)
    return before


def stage_inputs(replay, submissions, out):
    """Copy only owned snapshots; replay is strictly read-only."""
    srcs = {
        'vi50/fusion_context.json.zst': replay/'b660/vi50/fusion_context.json.zst',
        'vi50/d8w.json': replay/'derived/vi50/d8w.json',
        'vi51/cosines_vi51b.json': replay/'b660/vi51/cosines_vi51b.json',
        'vi55/q3e8b_full_top1000.json.zst': replay/'b660/vi55/q3e8b_full_top1000.json.zst',
        'vi55/all_ids.npy': replay/'b660/vi55/all_ids.npy',
        'vi45/bad_ids.json': replay/'b660/vi45/bad_ids.json',
        'vi45/new_ids.json': replay/'b660/vi45/new_ids.json',
        'vi45/links_corpus_vi_en_zh.parquet': replay/'b660/vi45/links_corpus_vi_en_zh.parquet',
        'vi53/submission_vi53.zip': submissions/'submission_vi53.zip',
        'vi56/submission_vi56.zip': submissions/'submission_vi56.zip',
        'vi56/new_ids.json': replay/'b660/vi56/new_ids.json',
        'vi56/expected_bge.json': replay/'b660/vi56/expected_bge.json',
        'vi56/expected_qwen.json': replay/'b660/vi56/expected_qwen.json',
    }
    modal = replay/'modal/<MODAL_PROFILE_D>/r2ai-vi56-data'
    srcs['vi56/prepare_report.json'] = modal/'prepare_report.json'
    for rel, target in [('in', 'modal_input'), ('out', 'outputs')]:
        for p in sorted((modal/rel).rglob('*')):
            if p.is_file():
                srcs['vi56/'+target+'/'+p.relative_to(modal/rel).as_posix()] = p
    audit = {}
    for rel, src in srcs.items():
        dst = out/'inputs'/rel
        audit[str(dst)] = copy_checked(src, dst)
    write(out/'input_audit.json', dict(old_inputs=audit))
    # Exact source code fingerprint makes reusing vi56 bench explicit.
    write(out/'gpu_code.json', dict(vi56=fingerprint(Path(vi56.__file__)),
          scorer_sha256=hashlib.sha256(inspect.getsource(vi56.load_scorer).encode()).hexdigest(),
          gpu_sha256=hashlib.sha256(inspect.getsource(vi56.gpu).encode()).hexdigest(),
          implementation='direct vi56.gpu; no scorer changes', bench='vi56 bench_*_fix'))


def selection(out):
    import numpy as np
    src = out/'inputs'
    top = read(src/'vi55/q3e8b_full_top1000.json.zst')
    pool = read(src/'vi50/fusion_context.json.zst')['data']['pool']
    bad = set(map(int, read(src/'vi45/bad_ids.json'))) - set(map(int, read(src/'vi45/new_ids.json')))
    prep = read(src/'vi56/prepare_report.json')
    empty = {int(d) for q, d in prep['empty_pairs']}
    if (out/'extra/prepare_report.json').exists():
        empty.update(int(d) for q, d in read(out/'extra/prepare_report.json')['empty_pairs'])
    allowed = set(map(int, np.load(src/'vi55/all_ids.npy'))) - bad - empty
    fresh = select(top, pool, allowed)
    top100 = select(top, pool, allowed, 100)
    old = read(src/'vi56/new_ids.json')
    old = {q: [d for d in ds if d in allowed] for q, ds in old.items()}
    assert old == top100, 'STOP vi56 selection not identical'
    extra = {q: [d for d in ds if d not in set(old[q])] for q, ds in fresh.items()}
    for path, obj in [('new_ids.json', fresh), ('top100_ids.json', old), ('extra/new_ids.json', extra)]:
        write(out/path, obj)
    q3e = {q: {str(d): float(s) for d, s in rows[:200] if int(d) in set(fresh[q])} for q, rows in top.items()}
    write(out/'q3e_new.json', q3e)
    write(out/'extra/q3e_new.json', {q: {str(d): q3e[q][str(d)] for d in ds} for q, ds in extra.items()})
    report = dict(top=200, new_pairs=sum(map(len, fresh.values())), vi56_pairs=sum(map(len, old.values())),
                  extra_pairs=sum(map(len, extra.values())), unique_docs=len({d for ds in fresh.values() for d in ds}),
                  bad_excluded=len(bad), empty_excluded=len(empty))
    write(out/'selection_report.json', report)
    print(json.dumps(report), flush=True)


def reuse(out):
    """vi56.aggregate verifies every complete input/output SHA and exact identity."""
    old = out/'inputs/vi56'
    top100 = read(out/'top100_ids.json')
    q3e = read(out/'q3e_new.json')
    write(old/'q3e_new.json', {q: {str(d): q3e[q][str(d)] for d in ds} for q, ds in top100.items()})
    # aggregate expects historical raw selection, including empty and excluded bad rows.
    raw = read(old/'new_ids.json')
    top = read(out/'inputs/vi55/q3e8b_full_top1000.json.zst')
    write(old/'q3e_new.json', {q: {str(d): float(s) for d, s in rows[:100] if int(d) in set(raw[q])}
                            for q, rows in top.items()})
    scores = vi56.aggregate(old)
    write(out/'reused_scores.json', scores)
    write(out/'reuse_report.json', dict(pairs=sum(map(len, top100.values())), exact_id_sha=True,
          parts=len(read(old/'prepare_report.json')['parts']), source='vi56 completed_part + aggregate'))
    return scores


def load_base(out):
    from vi41c import load_zip
    src = out/'inputs'
    path = src/'vi48/submission_vi48.zip'
    assert fingerprint(path)['sha256'] == '0bfc09df2e72b01b61bae571521f84ec005bed48b620ee0b536d710ae6f8281e'
    return sorted(load_zip(path), key=lambda r: r['id'])


def compute(out, fresh, scores, base):
    from vi52 import fused, doclists
    src = out/'inputs'
    ctx = read(src/'vi50/fusion_context.json.zst')
    lengths = read(src/'vi56/prepare_report.json')['lengths'].copy()
    if (out/'extra/prepare_report.json').exists():
        lengths.update(read(out/'extra/prepare_report.json')['lengths'])
    data, bge, q3e, d8w = vi56.extend(ctx, ctx['bge'], read(src/'vi51/cosines_vi51b.json'),
                                    read(src/'vi50/d8w.json'), fresh, scores, lengths)
    rankings = fused(data, bge, d8w, q3e)
    return doclists(base, rankings), rankings, (data, bge, d8w, q3e)


def sanity(out):
    from vi52 import fused, doclists, compare, load_docs
    src = out/'inputs'
    base = load_base(out)
    ctx = read(src/'vi50/fusion_context.json.zst')
    actual = doclists(base, fused(ctx['data'], ctx['bge'], read(src/'vi50/d8w.json'), read(src/'vi51/cosines_vi51b.json')))
    ref53 = load_docs(src/'vi53/submission_vi53.zip')
    first = compare(actual, ref53)
    first['mismatched_ids'] = [q for q in actual if actual[q] != ref53[q]]
    del ctx, actual
    scores = read(out/'reused_scores.json') if (out/'reused_scores.json').exists() else reuse(out)
    actual, _, _ = compute(out, read(out/'top100_ids.json'), scores, base)
    ref56 = load_docs(src/'vi56/submission_vi56.zip')
    second = compare(actual, ref56)
    second['mismatched_ids'] = [q for q in actual if actual[q] != ref56[q]]
    report = dict(disabled_new_vs_vi53=first, top100_vs_vi56=second)
    write(out/'sanity.json', report)
    second['set_and_count_exact'] = sum(set(actual[q]) == set(ref56[q]) and len(actual[q]) == len(ref56[q]) for q in actual)
    second['order_mismatches'] = 1200-second['exact']
    report['criteria'] = 'DECISIONS 04/11 15:05: vi53 exact order; vi56 set + count'
    write(out/'sanity.json', report)
    assert first['exact'] == second['set_and_count_exact'] == 1200, ('STOP before ZIP: sanity mismatch', report)
    print('SANITY '+json.dumps(report), flush=True)
    return report


def aggregate(out):
    old = read(out/'reused_scores.json')
    extra = vi56.aggregate(out/'extra')
    fresh = read(out/'new_ids.json')
    result = {}
    for q, ds in fresh.items():
        result[q] = {}
        for d in ds:
            sd = str(d)
            assert not (sd in old.get(q, {}) and sd in extra.get(q, {}))
            result[q][sd] = old.get(q, {}).get(sd, extra.get(q, {}).get(sd))
            assert set(result[q][sd]) == {'q3e','bge_doc','q8b_doc','bge_win','d8w'}
    write(out/'new_scores.json', result)
    return result


def cx_extra(out):
    src = out/'extra'
    dest = out/'cx_extra'
    for p in sorted((src/'modal_input').glob('qwen.part*')):
        copy_checked(p, dest/'in'/p.name)
    for p in sorted((src/'outputs').rglob('qwen*.scores.json')):
        assert vi56.completed_part(src/'modal_input'/p.name.removesuffix('.scores.json'), p)
        copy_checked(p, dest/'out'/p.parent.name/p.name)
        meta = p.with_suffix('.complete.json')
        copy_checked(meta, dest/'out'/p.parent.name/meta.name)
    manifest = {p.relative_to(dest).as_posix(): fingerprint(p) for p in sorted(dest.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'}
    assert manifest
    write(dest/'MANIFEST.json', dict(files=manifest, definition='vi56 format; q8b_doc and all U900 d8w; only ranks101-200'))
    zp = out/'cx_extra.zip'
    assert not zp.exists()
    with zipfile.ZipFile(zp, 'x', compression=zipfile.ZIP_STORED) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file(): z.write(p, p.relative_to(dest).as_posix())
    write(out/'cx_extra_report.json', fingerprint(zp))


def build(out):
    from vi41c import digest, load_zip
    from vi52 import load_docs, compare
    from vi50 import protect
    from vi29 import assert_clean, load_corpus_ids
    from reorder_zip import write_reordered_zip
    from vi23 import check_zip
    start = time.monotonic()
    zp, jp, intent = out/'submission_vi59d.zip', out/'submission_vi59d.json', out/'build_intent.json'
    assert not any(p.exists() for p in (zp, jp, intent)), 'ONE build only'
    import ctypes
    class MemoryStatus(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)] + [(k,ctypes.c_ulonglong) for k in
                    ('total_phys','avail_phys','total_page','avail_page','total_virtual','avail_virtual','extended')]
    mem=MemoryStatus();mem.length=ctypes.sizeof(mem)
    assert ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
    write(out/'build_resources.json',dict(available_ram_bytes=mem.avail_phys,required_ram_bytes=8*1024**3))
    assert mem.avail_phys>=8*1024**3,'WAIT: free RAM below8GiB; no build started'
    audit = read(out/'input_audit.json')['old_inputs']
    assert all(fingerprint(Path(p)) == m for p, m in audit.items())
    sanity_report = read(out/'sanity.json')
    assert sanity_report['disabled_new_vs_vi53']['exact'] == 1200
    assert sanity_report['top100_vs_vi56']['set_and_count_exact'] == 1200
    base = load_base(out)
    scores = aggregate(out)
    fresh = read(out/'new_ids.json')
    _, rankings, (data,bge,d8w,q3e) = compute(out, fresh, scores, base)
    rows = [protect(r, rankings[str(r['id'])])[0] for r in base]
    ref53 = load_docs(out/'inputs/vi53/submission_vi53.zip')
    ref56 = load_docs(out/'inputs/vi56/submission_vi56.zip')
    offline = vi56.offline(base, rows, ref53, rankings, data,bge,d8w,q3e,fresh)
    offline['vs_vi56'] = compare({r['id']:r['relevant_docs'] for r in rows}, ref56)
    offline['source_z_vs_vi56'] = vi56.offline(base,rows,ref56,rankings,data,bge,d8w,q3e,fresh)['source_z']
    extra = read(out/'extra/new_ids.json')
    import dwmax_diag as dw
    counts_extra = [len(set(r['relevant_docs']) & set(extra[str(r['id'])])) for r in rows]
    offline['extra101_200_in'] = dw.dist(counts_extra)
    offline['extra_zero_queries'] = sum(n == 0 for n in counts_extra)
    write(out/'offline.json', offline)
    src = out/'inputs/vi45'
    bad = set(map(int,read(src/'bad_ids.json'))) - set(map(int,read(src/'new_ids.json')))
    corpus = load_corpus_ids(src/'links_corpus_vi_en_zh.parquet')
    clean = assert_clean(rows,bad,corpus)
    chunks = {r['id']:digest(r['relevant_chunks']) for r in base}
    counts = {r['id']:len(r['relevant_docs']) for r in base}
    assert all(digest(r['relevant_chunks']) == chunks[r['id']] and len(r['relevant_docs']) == counts[r['id']] for r in rows)
    assert all(len(r['relevant_docs']) == len(set(r['relevant_docs'])) for r in rows)
    hashes = {r['id']:digest(r) for r in rows}
    del base,scores,rankings,data,bge,d8w,q3e
    gc.collect()
    with intent.open('x',encoding='utf-8') as f: f.write('{"build_attempts":1}')
    info = write_reordered_zip(rows,zp)
    assert zp.stat().st_size <= 99_000_000
    del rows
    gc.collect()
    try:
        with zipfile.ZipFile(zp) as z:
            assert z.namelist() == ['submission.json'] and z.infolist()[0].compress_type == zipfile.ZIP_LZMA
            assert z.testzip() is None
            with z.open('submission.json') as s, jp.open('xb') as d: shutil.copyfileobj(s,d)
        final = load_zip(zp)
        assert {r['id']:digest(r) for r in final} == hashes and set(hashes) == set(range(1,1201))
        assert all(digest(r['relevant_chunks']) == chunks[r['id']] and len(r['relevant_docs']) == counts[r['id']] for r in final)
        assert assert_clean(final,bad,corpus) == clean
        del final
        gc.collect()
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),'validate',str(jp)],check=True)
        assert check_zip(str(jp),str(zp),str(out/'check_zip.json'),'vi59d',max_zip=99_000_000) == 0
        assert all(fingerprint(Path(p)) == m for p, m in audit.items())
        write(out/'build_report.json', dict(**info,**fingerprint(zp),validate=0,check_zip=0,assert_clean=clean,
              sanity=sanity_report,ids=1200,chunks_equal_vi48=1200,doc_counts_equal_vi48=1200,
              entry='submission.json',compression='LZMA',build_attempts=1,seconds=time.monotonic()-start))
    finally:
        if jp.exists(): jp.unlink()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage',choices=['inputs','select','prepare','reuse','sanity','aggregate','cx-extra','build'])
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--root',type=Path,default=Path('D:/r2ai'))
    ap.add_argument('--replay',type=Path)
    ap.add_argument('--submissions',type=Path)
    a = ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    if a.stage == 'inputs': stage_inputs(a.replay,a.submissions,a.out)
    elif a.stage == 'select': selection(a.out)
    elif a.stage == 'prepare':
        from vi39 import wait_lock
        with wait_lock(a.root/'locks/heavy_job.lock',f'vi59 prepare pid={os.getpid()}'):
            vi56.prepare(a.root,a.out/'extra')
    else:
        {'reuse':reuse,'sanity':sanity,'aggregate':aggregate,'cx-extra':cx_extra,'build':build}[a.stage](a.out)


if __name__ == '__main__': main()
