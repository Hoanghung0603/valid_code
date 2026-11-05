"""Collect exact Nemotron pool, fixed fusion, one compression, full final checks."""
import argparse
import gc
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from vi48 import fingerprint,read,write
from vi54 import fused

TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))
OUT=TEMP/'vi54'


def collect():
    import dwmax_diag as dw
    import numpy as np
    prep=read(OUT/'prepare_report.json');pool=read(OUT/'modal_input/pool.json')
    scores={q:{} for q in pool};poolsets={q:set(ids) for q,ids in pool.items()}
    seen=set();files={};seen_docs=set();query=None;states=[]
    for p in OUT.glob('state_*.json'):
        s=read(p);states.append(s)
        for name,r in s['calls'].items():
            if r['kind']!='bulk':continue
            assert r['status']=='done' and r.get('collected')
            dest=OUT/'outputs'/s['profile']/name
            for n,meta in r['result']['files'].items():
                assert fingerprint(dest/n)==meta
                files[str(dest/n)]=meta
            qe=np.load(dest/'query_embeddings.npz'); qids=qe['ids'];qemb=qe['emb']
            assert set(map(int,qids))==set(range(1,1201)) and np.isfinite(qemb).all()
            assert np.max(abs(np.linalg.norm(qemb,axis=1)-1))<2e-4
            if query is None:query=(qids.copy(),qemb.copy())
            else:
                assert np.array_equal(qids,query[0])
                cos=np.sum(qemb*query[1],axis=1)
                assert cos.min()>=.999 and cos.mean()>=.9999
            for part in r['parts']:
                assert part not in seen;seen.add(part)
                rows=list(__import__('vi54').iterz(OUT/'modal_input'/part));ids=[r['id'] for r in rows]
                z=np.load(dest/(part+'.emb.npz'));emb=z['emb'].astype(np.float32)
                assert np.array_equal(z['ids'],np.asarray(ids)) and np.isfinite(emb).all()
                assert np.max(abs(np.linalg.norm(emb,axis=1)-1))<.002
                assert not seen_docs.intersection(ids);seen_docs.update(ids)
                part_scores=dw.read(dest/(part+'.cosines.json.zst'))
                for q,ds in part_scores.items():
                    assert not set(scores[q]).intersection(ds)
                    assert set(ds)=={str(d) for d in ids if d in poolsets[q]}
                    assert all(math.isfinite(v) for v in ds.values())
                    scores[q].update(ds)
    assert seen==set(prep['parts']) and len(seen_docs)==223065
    for q,ids in pool.items():assert set(scores[q])==set(map(str,ids))
    assert sum(map(len,scores.values()))==609662
    dw.write(OUT/'nemotron_cosines.json.zst',scores)
    write(OUT/'collect_report.json',dict(documents=len(seen_docs),pairs=609662,parts=29,
        exact_pool=True,embeddings_finite_unit=True,input_order=True,all_output_sha=True,
        cosine=fingerprint(OUT/'nemotron_cosines.json.zst'),files=files))
    print('COLLECT_AUDIT_PASS',flush=True)


def build():
    import dwmax_diag as dw
    from vi41c import digest,indexed,load_zip
    from vi50 import protect
    from vi29 import assert_clean,load_corpus_ids
    from vi52 import compare,doclists,load_docs,fused as qwen_fused
    from scipy.stats import spearmanr
    from reorder_zip import write_reordered_zip
    start=time.monotonic();zp=OUT/'submission_vi54.zip';jp=OUT/'submission_vi54.json'
    intent=OUT/'build_intent.json'
    assert not any(p.exists() for p in (intent,zp,jp,OUT/'build_report.json')),'ONE build only'
    paths=[TEMP/'vi50/fusion_context.json.zst',TEMP/'vi50/d8w.json',TEMP/'vi51/cosines_vi51b.json',
           OUT/'nemotron_cosines.json.zst',TEMP/'vi48/submission_vi48.zip',TEMP/'vi52/submission_vi52.zip']
    if (TEMP/'vi53/submission_vi53.zip').exists():paths.append(TEMP/'vi53/submission_vi53.zip')
    before={str(p):fingerprint(p) for p in paths}
    ctx=dw.read(paths[0]);data,bge=ctx['data'],ctx['bge'];d8w=read(paths[1]);q3e=read(paths[2]);nem=dw.read(paths[3])
    assert read(OUT/'collect_report.json')['exact_pool']
    for q,hits in data['pool'].items():
        assert set(q3e[q])==set(nem[q])=={str(h[0]) for h in hits}
        assert all(math.isfinite(v) for v in nem[q].values())
    sanity=fused(data,bge,d8w,q3e,{})==qwen_fused(data,bge,d8w,q3e)
    assert sanity
    base=sorted(load_zip(paths[4]),key=lambda r:r['id']);assert set(indexed(base))==set(range(1,1201))
    rankings=fused(data,bge,d8w,q3e,nem);rows=[];protected=0
    for r in base:
        row,m=protect(r,rankings[str(r['id'])]);rows.append(row);protected+=m
    lists={r['id']:r['relevant_docs'] for r in rows}
    offline=dict(without_nemotron_full_rankings_sanity=1200,protected_docs=protected)
    for label,p in [('vi52',paths[5]),('vi53',TEMP/'vi53/submission_vi53.zip')]:
        if p.exists():offline['vs_'+label]=compare(lists,load_docs(p));gc.collect()
    x=[];y=[];bx=[];by=[];macroq=[];macrob=[];overlap=[]
    for q,ds in nem.items():
        ids=list(ds);a=[ds[d] for d in ids];b=[q3e[q][d] for d in ids]
        x.extend(a);y.extend(b);macroq.append(float(spearmanr(a,b).statistic))
        dense={str(h[0]):float(h[2]) for h in data['dense'][q]};common=[d for d in ds if d in dense]
        c=[ds[d] for d in common];d=[dense[d] for d in common]
        bx.extend(c);by.extend(d);macrob.append(float(spearmanr(c,d).statistic))
        na=sorted(ids,key=lambda d:(-ds[d],int(d)))[:30]
        qa=sorted(ids,key=lambda d:(-q3e[q][d],int(d)))[:30]
        overlap.append(len(set(na)&set(qa)))
    offline.update(spearman_nemotron_q3e8b_pooled=float(spearmanr(x,y).statistic),
        spearman_nemotron_q3e8b_macro=sum(macroq)/len(macroq),
        spearman_nemotron_bgem3_pooled=float(spearmanr(bx,by).statistic),
        spearman_nemotron_bgem3_macro=sum(macrob)/len(macrob),
        top30_nemotron_q3e8b_overlap_mean=sum(overlap)/1200,correlation_bge_pairs=len(bx))
    write(OUT/'offline.json',offline)
    chunks={r['id']:digest(r['relevant_chunks']) for r in base};counts={r['id']:len(r['relevant_docs']) for r in base}
    source=TEMP/'vi45';bad=set(map(int,read(source/'bad_ids.json')))-set(map(int,read(source/'new_ids.json')))
    clean=assert_clean(rows,bad,load_corpus_ids(source/'links_corpus_vi_en_zh.parquet'))
    assert all(len(r['relevant_docs'])==len(set(r['relevant_docs'])) for r in rows)
    assert all(digest(r['relevant_chunks'])==chunks[r['id']] and len(r['relevant_docs'])==counts[r['id']] for r in rows)
    hashes={r['id']:digest(r) for r in rows}
    del base,ctx,data,bge,d8w,q3e,nem,rankings,x,y,bx,by;gc.collect()
    with intent.open('x',encoding='utf-8') as f:json.dump(dict(build_attempts=1),f)
    info=write_reordered_zip(rows,zp);assert zp.stat().st_size<=99_000_000
    del rows;gc.collect()
    try:
        with zipfile.ZipFile(zp) as z:
            assert z.namelist()==['submission.json'] and z.infolist()[0].compress_type==zipfile.ZIP_LZMA
            assert z.testzip() is None
            with z.open('submission.json') as src,jp.open('xb') as dst:shutil.copyfileobj(src,dst)
        final=load_zip(zp);assert {r['id']:digest(r) for r in final}==hashes
        assert all(digest(r['relevant_chunks'])==chunks[r['id']] and len(r['relevant_docs'])==counts[r['id']] for r in final)
        del final;gc.collect()
        subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),'validate',str(jp)],check=True)
        from vi23 import check_zip
        result=check_zip(str(jp),str(zp),str(OUT/'check_zip.json'),'vi54',max_zip=99_000_000);assert result==0
        assert all(fingerprint(Path(p))==m for p,m in before.items())
        report=dict(**info,**fingerprint(zp),validate=0,check_zip=0,assert_clean=clean,crc=0,ids=1200,
            chunks_equal_vi48=1200,doc_counts_equal_vi48=1200,no_duplicate_docs=True,entry='submission.json',
            compression='LZMA',build_attempts=1,inputs_unchanged=True,seconds=time.monotonic()-start)
        write(OUT/'build_report.json',report);print('BUILD_DONE '+json.dumps(report),flush=True)
    finally:
        if jp.exists():jp.unlink()


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['collect','build']);a=ap.parse_args()
    if a.stage=='collect':collect()
    else:
        from dwmax_diag import owned_lock
        with owned_lock(OUT/'build.lock','vi54 build'):build()
