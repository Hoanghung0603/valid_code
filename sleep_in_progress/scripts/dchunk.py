"""Dense-chunk 512/50 experiment. CPU export/build; no submission or vector download."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = Path(os.environ.get('R2AI_ROOT', 'D:/r2ai'))
OUT = ROOT / 'data/sub/dchunk'
MODEL = 'BAAI/bge-m3'


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def token_windows(ids, offsets, size=512, overlap=50):
    """Full coverage, exact token stride, no redundant tail; character offsets are verbatim.

    XLM-R can map two tokens to one Unicode character. Offsets may thus overlap at a
    boundary; never decode token IDs to recover text (which changes whitespace).
    """
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Require 0 <= overlap < size')
    if len(ids) != len(offsets):
        raise ValueError('IDs/offsets length mismatch')
    for start in range(0, len(ids), size-overlap):
        end = min(start+size, len(ids))
        a, b = offsets[start][0], offsets[end-1][1]
        if b <= a:
            raise ValueError('Empty/nonmonotone character span')
        yield start, end, a, b
        if end == len(ids):
            break


@contextmanager
def lock(stage):
    from dwmax_diag import owned_lock
    path = ROOT / 'locks/heavy_job.lock'
    deadline = time.monotonic()+120*60
    while path.exists():
        if time.monotonic() >= deadline:
            raise TimeoutError('Foreign heavy_job.lock remained 120 minutes')
        print('WAIT FOREIGN LOCK', path.read_text(encoding='utf-8'), flush=True)
        time.sleep(300)
    with owned_lock(path, f'dchunk {stage} pid={os.getpid()}'):
        yield


def inspect():
    p = ROOT / 'locks/heavy_job.lock'
    print('LOCK', p.read_text(encoding='utf-8') if p.exists() else 'absent')
    print('FILES', [(x.name, x.stat().st_size) for x in sorted(OUT.glob('*')) if x.is_file()])
    print('INPUT', [(x.name,x.stat().st_size) for x in sorted((OUT/'input').glob('*'))])
    for name in ('top1000.npz','submission_dchunk30.zip','export_report.json','audit_report.json','build_report.json'):
        p = OUT/name
        if p.exists():
            print('SHA256',name,p.stat().st_size,sha(p))
    for p in OUT.glob('*.log'):
        with p.open('rb') as f:
            f.seek(max(0, p.stat().st_size-3000))
            print(f.read().decode('utf-8', 'replace'))
    for root in (ROOT / 'hf', ROOT / 'work', ROOT / '.cache'):
        if root.exists():
            print('TOKENIZERS', [str(p) for p in root.rglob('tokenizer.json')][:25])


def build_status():
    p=ROOT/'locks/heavy_job.lock'
    result=dict(lock=p.read_text(encoding='utf-8') if p.exists() else None)
    for name in ('build_report.json','zip_report.json'):
        p=OUT/name
        if p.exists():
            result[name]=json.loads(p.read_text(encoding='utf-8'))
    p=OUT/'build.log'
    if p.exists():
        with p.open('rb') as f:
            f.seek(max(0,p.stat().st_size-5000))
            result['log_tail']=f.read().decode('utf-8','replace')
        result['log_mtime']=p.stat().st_mtime
    print(json.dumps(result,ensure_ascii=False))
    subprocess.run(['schtasks','/Query','/TN','r2ai_dchunk_build','/FO','LIST','/V'],check=False)


def export(tokenizer):
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from bigindex import BigIndex
    from boilerplate import BoilerplateFilter
    from export_docs import iter_all_texts
    from dwmax_diag import safe_index
    t0 = time.time()
    dest = OUT / 'input'
    dest.mkdir(parents=True, exist_ok=True)
    assert not (dest / 'manifest.json').exists(), 'Refuse overwrite completed export'
    tok = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
    assert tok.is_fast
    tok.save_pretrained(OUT / 'tokenizer')
    index = ROOT / 'data/index/vi36'
    safe_index(index)
    idx = BigIndex(index)
    bp = BoilerplateFilter.from_json((index / 'boilerplate.json').read_text(encoding='utf-8'))
    corpus = set(map(int, pq.read_table(ROOT / 'data/sub/diag19/links_corpus_vi_en_zh.parquet', columns=['id'])['id'].to_pylist()))
    wanted = {int(d): i for i, d in enumerate(idx.ids) if int(d) in corpus}
    assert len(wanted) >= 720000, (len(wanted), idx.n)
    chunks, docrows, files = [], [], []
    nchunk = ntoken = modeltokens = ndoc = empty = 0
    # Uniform deterministic reservoir independent of shard/domain/length ordering.
    import random
    rng, bench = random.Random(30512), []

    def flush():
        nonlocal chunks, docrows
        if not chunks:
            return
        n = len(files)
        fn = dest / f'tokens_{n:04d}.npz'
        lens = np.array([len(c[5]) for c in chunks], dtype=np.int32)
        ptr = np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lens)))
        np.savez_compressed(fn, ids=np.concatenate([np.asarray(c[5], dtype=np.int32) for c in chunks]),
                            ptr=ptr, chunk_id=np.array([c[0] for c in chunks], dtype=np.int64))
        meta = OUT / f'chunks_{n:04d}.parquet'
        pq.write_table(pa.Table.from_pylist([dict(chunk_id=c[0], doc_id=c[1], idx=c[2],
                                                 start=c[3], end=c[4], tokens=len(c[5])-2) for c in chunks]),
                       meta, compression='zstd')
        docfile = OUT / f'docs_{n:04d}.parquet'
        pq.write_table(pa.Table.from_pylist(docrows), docfile, compression='zstd')
        files.append(dict(name=fn.name, chunks=len(chunks), tokens=int(lens.sum()), bytes=fn.stat().st_size,
                          sha256=sha(fn), metadata=meta.name, documents=docfile.name))
        print('PART', files[-1], flush=True)
        chunks, docrows = [], []

    for d, i, text in iter_all_texts(idx, wanted, bp, sys.maxsize, no_title=False):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True, truncation=False,
                  verbose=False)
        ids, offsets = enc['input_ids'], enc['offset_mapping']
        ntoken += len(ids)
        ndoc += 1
        empty += not bool(ids)
        docrows.append(dict(doc_id=d, idx=i, text=text))
        for s, e, a, b in token_windows(ids, offsets):
            seq = tok.build_inputs_with_special_tokens(ids[s:e])
            assert len(seq) == e-s+2 and text[a:b]
            c = (nchunk, d, i, a, b, seq)
            chunks.append(c)
            nchunk += 1
            modeltokens += len(seq)
            if len(bench) < 4096:
                bench.append(c)
            else:
                j = rng.randrange(nchunk)
                if j < len(bench):
                    bench[j] = c
        if len(chunks) >= 32768:
            flush()
        if ndoc % 25000 == 0:
            print('EXPORT', ndoc, nchunk, ntoken, round(time.time()-t0), flush=True)
    flush()
    assert ndoc == len(wanted), (ndoc, len(wanted))
    lens = np.array([len(c[5]) for c in bench], dtype=np.int64)
    np.savez_compressed(dest / 'bench.npz', ids=np.concatenate([np.asarray(c[5], dtype=np.int32) for c in bench]),
                        ptr=np.concatenate((np.zeros(1, dtype=np.int64), np.cumsum(lens))),
                        chunk_id=np.array([c[0] for c in bench], dtype=np.int64))
    queries = pq.read_table(ROOT / 'data/query.parquet').to_pylist()
    assert len(queries) == 1200 and {q['id'] for q in queries} == set(range(1, 1201))
    write(dest / 'queries.json', queries)
    rep = dict(model=MODEL, tokenizer=tokenizer, content_tokens=512, overlap=50, stride=462,
               special_tokens=2, query_max_length=256, title='prepend once as dense original; no text cap',
               documents=ndoc, excluded_outside_corpus=idx.n-ndoc, empty_documents=empty,
               chunks=nchunk, source_tokens=ntoken, embedded_tokens_with_overlap_specials=modeltokens,
               elapsed=time.time()-t0, files=files, bench_chunks=len(bench), bench_tokens=int(lens.sum()),
               bytes=sum(x['bytes'] for x in files))
    write(dest / 'manifest.json', rep)
    write(OUT / 'export_report.json', rep)
    print('EXPORT DONE', json.dumps({k:v for k,v in rep.items() if k != 'files'}), flush=True)


def collapse(chunk_ids, scores, metadata):
    """Maximum chunk score, deterministic (score desc, chunk ID asc) tie-break."""
    best = {}
    import math
    for c, score in zip(chunk_ids, scores):
        c, score = int(c), float(score)
        if not math.isfinite(score):
            raise ValueError('Nonfinite score')
        m = metadata[c]
        d = int(m['doc_id'])
        row = [d, int(m['idx']), score, c]
        if d not in best or (-score, c) < (-best[d][2], best[d][3]):
            best[d] = row
    return sorted(best.values(), key=lambda r: (-r[2], r[3]))


def export_pool_scope():
    """Export only chunk IDs; keep immutable full-corpus token parts unchanged."""
    import numpy as np
    import pyarrow.parquet as pq
    from dwmax_diag import read
    pool = {int(r[0]) for rows in read(ROOT/'data/sub/vi36/in1/pool_v36.json').values() for r in rows}
    selected, parents, tokens = [], set(), 0
    for p in sorted(OUT.glob('chunks_*.parquet')):
        for r in pq.read_table(p).to_pylist():
            if r['doc_id'] in pool:
                selected.append(r['chunk_id'])
                parents.add(r['doc_id'])
                tokens += r['tokens']+2
    audit = read(OUT/'audit_report.json')['alternatives']
    assert len(parents)==audit['expanded_pool_documents'] and tokens==audit['expanded_pool_tokens']
    path = OUT/'pool_scope.npz'
    np.savez_compressed(path,chunk_ids=np.asarray(selected,dtype=np.int64))
    write(OUT/'pool_scope.json',dict(scope='expanded_pool',note='Dense-chunk trên pool mở rộng, KHÔNG phải toàn kho',
          documents=len(parents),chunks=len(selected),tokens=tokens,sha256=sha(path)))


def build():
    import numpy as np
    import pyarrow.parquet as pq
    import vi29
    from evaluate import validate
    from dwmax_diag import read
    from vi23 import check_zip
    bulk = read(OUT/'bulk_report.json')
    assert sha(OUT/'top1000.npz')==bulk['top1000_sha256']
    hits = np.load(OUT / 'top1000.npz')
    assert hits['chunk_ids'].shape == hits['scores'].shape == (1200, 1000)
    assert len(hits['qids'])==1200 and set(map(int, hits['qids'])) == set(range(1,1201))
    assert np.isfinite(hits['scores']).all() and (hits['chunk_ids']>=0).all()
    assert np.all(np.diff(hits['scores'],axis=1)<=0)
    assert all(len(set(map(int,row)))==1000 for row in hits['chunk_ids'])
    wanted = set(map(int, hits['chunk_ids'].ravel()))
    meta = {}
    for p in sorted(OUT.glob('chunks_*.parquet')):
        for r in pq.read_table(p).to_pylist():
            if r['chunk_id'] in wanted:
                meta[r['chunk_id']] = r
    assert set(meta) == wanted
    ranks = {str(int(q)): collapse(cs, ss, meta) for q, cs, ss in
             zip(hits['qids'], hits['chunk_ids'], hits['scores'])}
    corpus = vi29.load_corpus_ids(ROOT / 'data/sub/diag19/links_corpus_vi_en_zh.parquet')
    bad = set(map(int, read(ROOT / 'data/sub/diag28/bad_ids.json'))) - set(map(int, read(ROOT / 'data/sub/diag30/new_ids.json')))
    excluded_pairs=sum(r[0] in bad or r[0] not in corpus for rows in ranks.values() for r in rows)
    ranks={q:[r for r in rows if r[0] not in bad and r[0] in corpus] for q,rows in ranks.items()}
    assert min(map(len, ranks.values())) >= 200, 'Top1000 chunks insufficient for doc top200'
    wanted_docs = {r[0] for rows in ranks.values() for r in rows[:30]}
    texts = {}
    for p in sorted(OUT.glob('docs_*.parquet')):
        for r in pq.read_table(p).to_pylist():
            if r['doc_id'] in wanted_docs:
                texts[r['doc_id']] = r['text']
    sub = []
    for q, rows in sorted(ranks.items(), key=lambda v:int(v[0])):
        chunks = []
        for d, i, score, c in rows[:30]:
            m = meta[c]
            assert m['doc_id']==d and m['idx']==i
            assert 0<=m['start']<m['end']<=len(texts[d])
            text = texts[d][m['start']:m['end']]
            assert text and text in texts[d]
            chunks.append(dict(doc_id=d, chunk_text=text))
        sub.append(dict(id=int(q), relevant_docs=[r[0] for r in rows[:30]], relevant_chunks=chunks))
    assert not validate(sub)
    clean = vi29.assert_clean(sub, bad, corpus)
    jp, zp = OUT / 'submission_dchunk30.json', OUT / 'submission_dchunk30.zip'
    jp.write_text(json.dumps(sub, ensure_ascii=False, separators=(',',':')), encoding='utf-8')
    subprocess.run([sys.executable, str(ROOT / 'scripts/evaluate.py'), 'validate', str(jp)], check=True)
    with zipfile.ZipFile(zp, 'w', compression=zipfile.ZIP_LZMA) as z:
        z.write(jp, 'submission.json')
    with zipfile.ZipFile(zp) as z:
        assert z.namelist() == ['submission.json'] and z.infolist()[0].compress_type == zipfile.ZIP_LZMA
        assert json.loads(z.read('submission.json')) == sub
    zcheck = check_zip(str(jp), str(zp), str(OUT / 'zip_report.json'), 'dchunk30', max_zip=99000000)
    assert zcheck == 0
    assert zp.stat().st_size < 99000000
    write(OUT / 'dense_chunk_doc_ranks.json', ranks)
    dense = read(ROOT / 'data/sub/vi36/in1/dense_v36.json')
    fhyde = read(ROOT / 'data/sub/vi37/cache_fhyde_v37.json')
    pool = read(ROOT / 'data/sub/vi36/in1/pool_v36.json')
    def ov(cache):
        return sum(len({r[0] for r in rows[:30]} & {int(r[0]) for r in cache[q][:30]}) for q,rows in ranks.items())/1200
    outside = [len({r[0] for r in rows[:200]} - {int(r[0]) for r in pool[q]}) for q,rows in ranks.items()]
    rep = dict(scope=bulk['scope'],scope_note=bulk['scope_note'], embedded_documents=bulk['documents'],
                zip_bytes=zp.stat().st_size, sha256=sha(zp), validate_exit=0, check_zip=zcheck, excluded_bad_or_outside_pairs=excluded_pairs,
                assert_clean=clean, verbatim_bad=0, queries=len(sub),zip_entries=1,zip_compression='LZMA',
                top1000_sha256=bulk['top1000_sha256'],top30_overlap_dense=ov(dense), top30_overlap_fhyde_vi37=ov(fhyde),
               top200_outside_vi37_pool_total=sum(outside), top200_outside_vi37_pool_mean=sum(outside)/1200,
                min_unique_docs_in_top1000=min(map(len,ranks.values())), note='Doc ranks censored beyond top1000 chunk; top200 certified by >=200 distinct parents; ties at cutoff may remain ambiguous')
    write(OUT / 'build_report.json', rep)
    print('BUILD DONE', json.dumps(rep), flush=True)


def check_sample(doc, actual, ids, ptr, tok):
    """Retokenize one complete document; compare every exported ID and offset."""
    enc = tok(doc['text'], add_special_tokens=False, return_offsets_mapping=True,
              truncation=False, verbose=False)
    expected = list(token_windows(enc['input_ids'], enc['offset_mapping']))
    assert len(expected) == len(actual)
    for (s,e,a,b),(i,r) in zip(expected,actual):
        assert (r['start'],r['end'],r['tokens']) == (a,b,e-s)
        assert doc['text'][a:b]
        seq = [tok.cls_token_id]+enc['input_ids'][s:e]+[tok.sep_token_id]
        assert seq == ids[ptr[i]:ptr[i+1]].tolist()
    return len(enc['input_ids'])


def audit(sample=2000, seed=30512, max_seconds=900):
    """Full SHA/count/structure audit, uniform fixed-seed document tokenizer sample.

    Each NPZ array is decompressed ONCE per part, never once per chunk. Tokenizer
    evidence is sampled, not a claim of independent full-corpus tokenization.
    """
    import numpy as np
    import pyarrow.parquet as pq
    import random
    from transformers import AutoTokenizer
    t0 = time.monotonic()
    tok = AutoTokenizer.from_pretrained(OUT / 'tokenizer')
    mp = OUT/'export_report.json'
    manifest = json.loads(mp.read_text(encoding='utf-8'))
    other = OUT/'input/manifest.json'
    if other.exists():
        assert sha(mp) == sha(other)
    selected = set(random.Random(seed).sample(range(manifest['documents']), min(sample,manifest['documents'])))
    count = docs = source_tokens = model_tokens = 0
    sampled = sample_tokens = empty = 0
    cap1 = cap2 = pool_tokens = pool_docs = 0
    poolpath = ROOT/'data/sub/vi36/in1/pool_v36.json'
    pool = {int(r[0]) for rows in json.loads(poolpath.read_text(encoding='utf-8')).values() for r in rows} if poolpath.exists() else set()
    seen = set()
    for part in manifest['files']:
        assert time.monotonic()-t0 < max_seconds, 'Sample audit exceeded 15 minute budget'
        zp = OUT/'input'/part['name']
        assert sha(zp)==part['sha256'] and zp.stat().st_size==part['bytes']
        with np.load(zp) as z:
            ids, ptr, cids = z['ids'], z['ptr'], z['chunk_id']
        rows = pq.read_table(OUT/part['metadata']).to_pylist()
        parents = pq.read_table(OUT/part['documents']).to_pylist()
        assert len(rows)==part['chunks']==len(cids) and len(ptr)==len(rows)+1
        assert ptr[0]==0 and ptr[-1]==len(ids)==part['tokens']
        assert np.array_equal(cids,np.arange(count,count+len(rows)))
        lengths = np.diff(ptr)
        assert lengths.min()>=3 and lengths.max()<=514
        assert np.all(ids[ptr[:-1]]==tok.cls_token_id) and np.all(ids[ptr[1:]-1]==tok.sep_token_id)
        bydoc = {}
        for i,r in enumerate(rows):
            assert r['chunk_id']==count+i==int(cids[i])
            assert r['tokens']==int(lengths[i])-2
            bydoc.setdefault(r['doc_id'],[]).append((i,r))
        assert set(bydoc) <= {d['doc_id'] for d in parents}
        for doc in parents:
            assert doc['doc_id'] not in seen
            seen.add(doc['doc_id'])
            actual = bydoc.get(doc['doc_id'],[])
            total = sum(r['tokens'] for _,r in actual)-50*max(0,len(actual)-1)
            for j,(i,r) in enumerate(actual):
                assert r['idx']==doc['idx'] and 0<=r['start']<r['end']<=len(doc['text'])
                assert j==len(actual)-1 or r['tokens']==512
                if j:
                    assert r['start']>=actual[j-1][1]['start'] and r['end']>=actual[j-1][1]['end']
            if docs in selected:
                assert check_sample(doc,actual,ids,ptr,tok)==total
                sampled += 1
                sample_tokens += total
            empty += not bool(actual)
            source_tokens += total
            cap1 += sum(r['tokens']+2 for _,r in actual[:1])
            cap2 += sum(r['tokens']+2 for _,r in actual[:2])
            if doc['doc_id'] in pool:
                pool_docs += 1
                pool_tokens += sum(r['tokens']+2 for _,r in actual)
            docs += 1
        count += len(rows)
        model_tokens += len(ids)
        print('AUDIT SAMPLE',count,docs,sampled,round(time.monotonic()-t0,2),flush=True)
    assert (count,docs,source_tokens,model_tokens)==(manifest['chunks'],manifest['documents'],manifest['source_tokens'],manifest['embedded_tokens_with_overlap_specials'])
    assert sampled==len(selected) and empty==manifest['empty_documents']
    rep = dict(chunks=count,documents=docs,source_tokens=source_tokens,embedded_tokens=model_tokens,
               mode='full SHA/count/structure; sampled independent tokenizer', seed=seed,
               sampled_documents=sampled,sampled_source_tokens=sample_tokens,
               count_errors=0,span_errors=0,sampled_token_id_errors=0,sha_errors=0,
               manifest_sha256=sha(mp),elapsed_seconds=time.monotonic()-t0,
               alternatives=dict(first_chunk_tokens=cap1,first_two_chunk_tokens=cap2,
                                 expanded_pool_documents=pool_docs,expanded_pool_tokens=pool_tokens))
    write(OUT/'audit_report.json',rep)
    print(json.dumps(rep),flush=True)


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['inspect','export','build','audit','pool-scope','build-status'])
    ap.add_argument('--tokenizer', default=MODEL)
    ap.add_argument('--sample', type=int, default=2000)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == 'inspect':
        return inspect()
    if a.stage == 'build-status':
        return build_status()
    with lock(a.stage):
        if a.stage == 'export':
            export(a.tokenizer)
        elif a.stage == 'audit':
            audit(sample=a.sample)
        elif a.stage == 'pool-scope':
            export_pool_scope()
        else:
            build()


if __name__ == '__main__':
    main()
