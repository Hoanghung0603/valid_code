"""vi50: fixed 8B U900 MaxP source, fixed vi48 chunks/counts; never submit."""
import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

from vi45 import cid
from vi48 import fingerprint, read, write

TEMP = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))
RATE = 4.5232/3600


def fused(data, bge, d8w=None):
    from fuse3 import fuse3
    from vi37 import docwin_hits
    extras = [(docwin_hits(data['pool'], bge), 1.)]
    if d8w is not None:
        extras.append((docwin_hits(data['pool'], d8w), 1.))
    return fuse3(data['pool'], data['q8b'], data['dense'], (1.,2.,2.,1.),60,data['hyde'],extras)


def protect(base, hits):
    from vi41c import merge_row
    candidate = dict(id=base['id'], relevant_docs=[int(h[0]) for h in hits[:len(base['relevant_docs'])]])
    row, missing, short = merge_row(base, candidate)
    assert short == 0 and len(row['relevant_docs']) == len(base['relevant_docs'])
    return row, missing


def target_docs(rankings, counts):
    return {q: [int(h[0]) for h in rows[:counts[q]+100]] for q, rows in rankings.items()}


def forecast(n, jobs=12):
    # Same measured vi45 throughput; every unseen prompt upper-bounded at 1024 tokens.
    seconds = n*1024/40661.696*1.3 + jobs*65
    return dict(new_pairs=n, gpu_seconds_upper=seconds, usd_upper=seconds*RATE+.3)


def prepare(root, out):
    import zstandard as zstd
    import dwmax_diag as dw
    import vi37
    from chunking import chunk_document
    from vi41c import load_zip
    start=time.monotonic()
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'prepare_report.json').exists()
    data={s:dw.read(root/f'data/sub/vi41b/in1/{s}_v41.json') for s in ('pool','q8b','dense','hyde')}
    bge=dw.read(root/'data/sub/vi41b/docwin_max_v41.json')
    rankings=fused(data,bge)
    assert rankings==dw.read(root/'data/sub/vi41b/cache_fhyde_v41.json')
    # vi48 n_q is exactly vi37's count; vi41c/vi47/vi48 only protect chunk docs.
    ref=dw.read(root/'data/sub/vi37/submission_vi37.json')
    counts={str(r['id']):len(r['relevant_docs']) for r in ref}
    del ref
    targets=target_docs(rankings,counts)
    reverse=defaultdict(list)
    for q, ds in targets.items():
        for d in ds:
            reverse[d].append(q)
    old={}
    for p in sorted((root/'data/sub/vi45/scores').glob('*.logits.json')):
        for r in dw.read(p):
            assert r['cid'] not in old and math.isfinite(r['logit8c'])
            old[r['cid']]=r
    assert len(old)==317601
    texts=[Path(p) for p in dw.read(root/'data/sub/vi41b/preverify_report.json')['files']]
    qt=vi37.query_texts()
    # Stream frozen whole-pool text; keep no multi-GB full-text dictionary.
    def rows():
        seen=set()
        for p in texts:
            with zstd.open(p,'rt',encoding='utf-8') as f:
                for line in f:
                    doc=json.loads(line); d=int(doc['id'])
                    if d not in reverse:
                        continue
                    assert d not in seen
                    seen.add(d)
                    chunks=chunk_document(doc['text'],'vi',max_units=900)
                    for q in reverse[d]:
                        yield q,d,chunks
        assert seen==set(reverse), ('missing target text',len(set(reverse)-seen))
    all_new=all_pairs=reused=0
    for q,d,chunks in rows():
        identities={cid(q,dict(doc_id=d,chunk_text=c.text)) for c in chunks}
        all_pairs+=len(identities)
        reused+=sum(c in old for c in identities)
        all_new+=sum(c not in old for c in identities)
    full_forecast=forecast(all_new)
    top8=full_forecast['usd_upper']>40
    dw.write(out/'full_forecast.json',dict(**full_forecast,all_pairs=all_pairs,reused_pairs=reused,top8=top8))
    print('FULL_FORECAST '+json.dumps(dw.read(out/'full_forecast.json')),flush=True)
    wins={}
    if top8:
        # Exact vi37 provenance owners; vi41 new documents use their fresh windows.
        owners=dw.read(root/'data/sub/diag37/docwin_max_provenance.json')
        layers=[('v37',root/'data/sub/vi37/docwin_new.json.zst')]
        layers += [(name,root/'data/sub'/rel) for name,rel,_ in dw.layer_specs()]
        for name,p in layers:
            obj=dw.read(p)
            for d, qs in reverse.items():
                for q in qs:
                    sd=str(d)
                    if sd in obj.get(q,{}) and (name=='v37' or owners.get(q,{}).get(sd)==name):
                        if sd not in wins.get(q,{}):
                            wins.setdefault(q,{})[sd]=obj[q][sd]
            del obj
        obj=dw.read(root/'data/sub/vi41b/modal_out/docwin_new.json.zst')
        for q, ds in obj.items():
            for d, ws in ds.items():
                if int(d) in reverse and q in reverse[int(d)]:
                    wins.setdefault(q,{})[d]=ws
        del obj,owners
    writer=vi37.Parts(out/'modal_input','pairs50')
    expected=[]; reused_rows=[]; empty=[]; paircount=newcount=0
    for q,d,chunks in rows():
        if top8 and len(chunks)>8:
            from chunks_from_docwin import chunk_score
            ws=wins.get(q,{}).get(str(d))
            assert ws, ('missing bge windows for top8',q,d)
            chunks=sorted(enumerate(chunks),key=lambda ic:(-chunk_score(ws,ic[1].start,ic[1].end),ic[0]))[:8]
            chunks=[c for _,c in chunks]
        if not chunks:
            empty.append([q,d])
        seen=set()
        for c in chunks:
            identity=cid(q,dict(doc_id=d,chunk_text=c.text))
            if identity in seen:
                continue
            seen.add(identity)
            expected.append(dict(qid=q,doc_id=d,cid=identity))
            paircount+=1
            if identity in old:
                r=old[identity]
                assert str(r['qid'])==q and int(r['doc_id'])==d
                reused_rows.append(r)
            else:
                writer.add(dict(qid=q,doc_id=d,cid=identity,query=qt[q],text=c.text))
                newcount+=1
    writer.close()
    assert forecast(newcount)['usd_upper']<=40
    vi37.writez(out/'expected.json.zst',expected)
    vi37.writez(out/'reused_logits.json.zst',reused_rows)
    vi37.writez(out/'fusion_context.json.zst',dict(data=data,bge=bge,rankings=rankings,targets=targets,counts=counts))
    files={}; counts_part={}
    for p in writer.files:
        files[p.name]=fingerprint(p)
        with zstd.open(p,'rt',encoding='utf-8') as f:
            counts_part[p.name]=sum(1 for line in f if line.strip())
    report=dict(queries=1200,target_query_docs=sum(map(len,targets.values())),target_unique_docs=len(reverse),
        pairs=paircount,new_pairs=newcount,reused_pairs=len(reused_rows),empty_query_docs=empty,
        top8=top8,full_forecast=full_forecast,forecast=forecast(newcount),files=files,counts=counts_part,
        source='vi41b underlying full-pool doc ranking used by vi41c; n_q=vi48=vi37',seconds=time.monotonic()-start)
    dw.write(out/'prepare_report.json',report)
    print('PREPARE_DONE '+json.dumps(report),flush=True)


def build(out):
    import dwmax_diag as dw
    from vi41c import load_zip,digest,indexed
    from vi29 import assert_clean,load_corpus_ids
    from reorder_zip import write_reordered_zip
    from scipy.stats import spearmanr
    start=time.monotonic()
    zp=out/'submission_vi50.zip'; jp=out/'submission_vi50.json'
    assert not zp.exists() and not (out/'build_report.json').exists(), 'ONE build only'
    input_paths=[out/name for name in ('prepare_report.json','fusion_context.json.zst','expected.json.zst','reused_logits.json.zst')]
    input_paths += [TEMP/'vi48/submission_vi48.zip']
    before={str(p):fingerprint(p) for p in input_paths}
    prep=read(out/'prepare_report.json')
    ctx=dw.read(out/'fusion_context.json.zst')
    expected={r['cid']:(str(r['qid']),str(r['doc_id'])) for r in dw.read(out/'expected.json.zst')}
    assert len(expected)==prep['pairs']
    scores=dw.read(out/'reused_logits.json.zst')
    newfiles=sorted((out/'outputs').rglob('*.logits.json'))
    for p in newfiles:
        before[str(p)]=fingerprint(p)
    new=[]
    for p in newfiles:
        new.extend(read(p))
    assert len(new)==prep['new_pairs']
    # Keep new logit values separate from immutable vi45.
    import vi37
    vi37.writez(out/'new_logits_vi50.json.zst',new)
    scores.extend(new)
    seen=set(); d8w={}
    for r in scores:
        q,d=str(r['qid']),str(r['doc_id'])
        assert expected[r['cid']]==(q,d) and r['cid'] not in seen and math.isfinite(r['logit8c'])
        seen.add(r['cid'])
        d8w.setdefault(q,{})[d]=max(d8w.get(q,{}).get(d,-math.inf),r['logit8c'])
    assert seen==set(expected)
    write(out/'d8w.json',d8w)
    rankings=fused(ctx['data'],ctx['bge'],d8w)
    base=sorted(load_zip(TEMP/'vi48/submission_vi48.zip'),key=lambda r:r['id'])
    assert len(base)==1200 and set(indexed(base))==set(range(1,1201))
    basehash={r['id']:digest(r['relevant_chunks']) for r in base}
    outrows=[]; changes=[]; protected=0
    for r in base:
        q=str(r['id'])
        assert len(r['relevant_docs'])==ctx['counts'][q]
        row,m=protect(r,rankings[q]); protected+=m
        a,b=set(row['relevant_docs']),set(r['relevant_docs'])
        changes.append(dict(qid=q,entering=sorted(a-b),leaving=sorted(b-a),count=len(a-b)))
        outrows.append(row)
    del base
    x=[]; bg=[]; fh=[]; macbg=[]; macfh=[]
    for q,ds in d8w.items():
        old={str(h[0]):h[2] for h in ctx['rankings'][q]}
        a=[]; b=[]; c=[]
        for d,v in ds.items():
            if d in ctx['bge'].get(q,{}) and d in old:
                a.append(v); b.append(ctx['bge'][q][d]); c.append(old[d])
        x.extend(a); bg.extend(b); fh.extend(c)
        for dest,other in ((macbg,b),(macfh,c)):
            rho=float(spearmanr(a,other).statistic)
            if math.isfinite(rho): dest.append(rho)
    offline=dict(new_pairs=prep['new_pairs'],reused_pairs=prep['reused_pairs'],top8=prep['top8'],
        changed_queries=sum(r['count']>0 for r in changes),doc_in_total=sum(r['count'] for r in changes),
        doc_out_total=sum(len(r['leaving']) for r in changes),doc_in_per_query=statistics.mean(r['count'] for r in changes),
        spearman_d8w_bge_pooled=float(spearmanr(x,bg).statistic),spearman_d8w_fhyde_pooled=float(spearmanr(x,fh).statistic),
        spearman_d8w_bge_macro=statistics.mean(macbg),spearman_d8w_fhyde_macro=statistics.mean(macfh),
        correlation_pairs=len(x),protected_docs=protected,per_query=changes)
    write(out/'offline.json',offline)
    source=TEMP/'vi45'
    bad=set(map(int,read(source/'bad_ids.json')))-set(map(int,read(source/'new_ids.json')))
    clean=assert_clean(outrows,bad,load_corpus_ids(source/'links_corpus_vi_en_zh.parquet'))
    assert all(digest(r['relevant_chunks'])==basehash[r['id']] for r in outrows)
    assert all(len(r['relevant_docs'])==ctx['counts'][str(r['id'])] for r in outrows)
    hashes={r['id']:digest(r) for r in outrows}
    del ctx,scores,new,expected,seen
    gc.collect()
    info=write_reordered_zip(outrows,zp)  # Exactly ONE compression/build.
    assert zp.stat().st_size<=99_000_000
    del outrows
    gc.collect()
    with zipfile.ZipFile(zp) as z:
        assert z.namelist()==['submission.json'] and z.infolist()[0].compress_type==zipfile.ZIP_LZMA
        assert z.testzip() is None
        with z.open('submission.json') as src,jp.open('xb') as dst:
            shutil.copyfileobj(src,dst)
    final=load_zip(zp)
    assert {r['id']:digest(r) for r in final}==hashes
    assert all(digest(r['relevant_chunks'])==basehash[r['id']] for r in final)
    del final
    gc.collect()
    subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),'validate',str(jp)],check=True)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'check-zip','--out',str(out)],check=True)
    assert all(fingerprint(p)==meta for p,meta in before.items())
    report=dict(**info,**fingerprint(zp),validate=0,check_zip=0,assert_clean=clean,crc=0,ids=1200,
        chunks_equal_vi48=1200,doc_counts_equal_vi48=1200,entry='submission.json',compression='LZMA',
        scored_pairs_new=prep['new_pairs'],reused_pairs=prep['reused_pairs'],build_attempts=1,
        input_fingerprints=before,inputs_unchanged=True,seconds=time.monotonic()-start)
    write(out/'build_report.json',report)
    print('BUILD_DONE '+json.dumps(report),flush=True)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['prepare','build','check-zip'])
    ap.add_argument('--root',type=Path,default=Path('D:/r2ai'))
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args()
    if a.stage=='check-zip':
        from vi23 import check_zip
        raise SystemExit(check_zip(str(a.out/'submission_vi50.json'),str(a.out/'submission_vi50.zip'),str(a.out/'check_zip.json'),'vi50',max_zip=99_000_000))
    from dwmax_diag import owned_lock
    lock=a.root/'locks/heavy_job.lock' if a.stage=='prepare' else a.out/'build.lock'
    with owned_lock(lock,f'vi50 {a.stage} pid={os.getpid()}'):
        prepare(a.root,a.out) if a.stage=='prepare' else build(a.out)


if __name__=='__main__':
    main()
