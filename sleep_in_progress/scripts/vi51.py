"""vi51: pinned Qwen3 embeddings, full vi41c pool, fixed k60 x1 and vi48 chunks."""
import argparse
from collections import defaultdict
import gc
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

from vi48 import fingerprint, read, write

MODELS = {
    'a': ('Qwen/Qwen3-Embedding-0.6B', '97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3'),
    'b': ('Qwen/Qwen3-Embedding-8B', '1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'),
}
RELEASE = '2025-06-05'  # model-card-linked official Qwen release blog
INSTRUCT = 'Given a Vietnamese medical question, retrieve web documents that answer the question'
RATE = 4.5232/3600
TEMP = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))


def query_text(q):
    return f'Instruct: {INSTRUCT}\nQuery:{q}'


def fused(data, bge, q3e=None):
    from fuse3 import fuse3
    from vi37 import docwin_hits
    extras = [(docwin_hits(data['pool'], bge), 1.)]
    if q3e is not None:
        extras.append((docwin_hits(data['pool'], q3e), 1.))
    return fuse3(data['pool'], data['q8b'], data['dense'], (1.,2.,2.,1.),60,data['hyde'],extras)


def protect(base, hits):
    from vi41c import merge_row
    candidate = dict(id=base['id'], relevant_docs=[int(h[0]) for h in hits[:len(base['relevant_docs'])]])
    row, missing, short = merge_row(base, candidate)
    assert short == 0 and len(row['relevant_docs']) == len(base['relevant_docs'])
    return row, missing


def iterz(path):
    import zstandard as zstd
    with zstd.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def prepare(root, out):
    """Re-export title + clean body, token-truncate BEFORE any workspace upload."""
    import random
    import zstandard as zstd
    from transformers import AutoTokenizer
    import dwmax_diag as dw
    from bigindex import BigIndex
    from boilerplate import BoilerplateFilter
    from export_docs import iter_all_texts
    from vi37 import Parts, writez
    import pyarrow.parquet as pq
    start=time.monotonic()
    out.mkdir(parents=True,exist_ok=True)
    assert not (out/'prepare_report.json').exists()
    data={s:dw.read(root/f'data/sub/vi41b/in1/{s}_v41.json') for s in ('pool','q8b','dense','hyde')}
    bge=dw.read(root/'data/sub/vi41b/docwin_max_v41.json')
    rankings=fused(data,bge)
    assert rankings==dw.read(root/'data/sub/vi41b/cache_fhyde_v41.json')
    writez(out/'fusion_context.json.zst',dict(data=data,bge=bge,rankings=rankings))
    wanted={int(h[0]):int(h[1]) for rows in data['pool'].values() for h in rows}
    pool={q:[int(h[0]) for h in rows] for q,rows in data['pool'].items()}
    (out/'modal_input').mkdir(exist_ok=True)
    write(out/'modal_input/pool.json',pool)
    tok=AutoTokenizer.from_pretrained(MODELS['a'][0],revision=MODELS['a'][1],padding_side='left')
    tokb=AutoTokenizer.from_pretrained(MODELS['b'][0],revision=MODELS['b'][1],padding_side='left')
    # Core BPE vocab/merges match, but added special tokens differ between revisions.
    # Encode each tier independently rather than assume interchangeable token IDs.
    qs=pq.read_table(root/'data/query.parquet').to_pylist()
    queries=[];queriesb=[]
    for q in qs:
        text=query_text(q['query'])
        ids=tok(text,truncation=True,max_length=256)['input_ids']
        queries.append(dict(id=int(q['id']),input_ids=ids))
        queriesb.append(dict(id=int(q['id']),input_ids=tokb(text,truncation=True,max_length=256)['input_ids']))
    write(out/'modal_input/queries.json',queries)
    write(out/'modal_input/queries_b.json',queriesb)
    write(out/'queries.jsonl.meta.json',{'queries':len(queries)})
    index=root/'data/index/vi36'
    dw.safe_index(index)
    idx=BigIndex(index)
    # Pool retains historical indices from earlier corpora; resolve by immutable doc_id.
    positions={int(d):i for i,d in enumerate(idx.ids)}
    assert set(wanted)<=set(positions)
    wanted={d:positions[d] for d in wanted}
    bp=BoilerplateFilter.from_json((index/'boilerplate.json').read_text(encoding='utf-8'))
    writer=Parts(out/'modal_input','docs51')
    writerb=Parts(out/'modal_input','docs51b')
    sample_ids=set(random.Random(5151).sample(sorted(wanted),min(2000,len(wanted))))
    bench=[];benchb=[];seen=set();tokens=0;tokensb=0;lengths={};batch=[]
    def flush():
        nonlocal tokens,tokensb,batch
        if not batch: return
        texts=[r[2] for r in batch]
        enc=tok(texts,truncation=True,max_length=2048)['input_ids']
        encb=tokb(texts,truncation=True,max_length=2048)['input_ids']
        for (d,i,text),ids,idsb in zip(batch,enc,encb):
            assert d not in seen and ids
            seen.add(d);tokens+=len(ids);lengths[str(d)]=len(ids)
            row=dict(id=d,input_ids=ids)
            writer.add(row)
            writerb.add(dict(id=d,input_ids=idsb));tokensb+=len(idsb)
            if d in sample_ids: bench.append(row)
            if d in sample_ids: benchb.append(dict(id=d,input_ids=idsb))
        batch=[]
    # No additional character cut: index pipeline clean + title, then exactly 2048 model tokens.
    for d,i,text in iter_all_texts(idx,wanted,bp,2**31-1):
        batch.append((d,i,text))
        if len(batch)>=128: flush()
    flush();writer.close();writerb.close()
    assert seen==set(wanted) and {r['id'] for r in bench}==sample_ids
    with zstd.open(out/'modal_input/bench.jsonl.zst','wt',encoding='utf-8') as f:
        for r in bench: f.write(json.dumps(r)+'\n')
    with zstd.open(out/'modal_input/bench_b.jsonl.zst','wt',encoding='utf-8') as f:
        for r in benchb: f.write(json.dumps(r)+'\n')
    write(out/'doc_lengths.json',lengths)
    parts={p.name:dict(**fingerprint(p),docs=sum(1 for _ in iterz(p)),tokens=sum(len(r['input_ids']) for r in iterz(p))) for p in writer.files}
    partsb={p.name:dict(**fingerprint(p),docs=sum(1 for _ in iterz(p)),tokens=sum(len(r['input_ids']) for r in iterz(p))) for p in writerb.files}
    files={p.name:fingerprint(p) for p in (out/'modal_input').iterdir()}
    report=dict(documents=len(seen),pairs=sum(map(len,pool.values())),queries=len(qs),tokens=tokens,
        query_tokens=sum(len(r['input_ids']) for r in queries),bench_docs=len(bench),
        bench_tokens=sum(len(r['input_ids']) for r in bench),parts=parts,parts_b=partsb,tokens_b=tokensb,files=files,models=MODELS,
        tokenizers_independently_applied=True,release=RELEASE,seconds=time.monotonic()-start)
    write(out/'prepare_report.json',report)
    print('PREPARE_DONE '+json.dumps(report),flush=True)


def token_batches(rows, budget):
    """Sort by length; budget padded tokens (stricter than sum of real tokens)."""
    order=sorted(range(len(rows)),key=lambda i:(len(rows[i]['input_ids']),i))
    batch=[]
    for i in order:
        if batch and len(rows[i]['input_ids'])*(len(batch)+1)>budget:
            yield batch;batch=[]
        batch.append(i)
    if batch:yield batch


def repeat_pass(minimum, mean):
    return minimum >= .999 and mean >= .9999


def gpu(tier, mode, source, dest, parts, watchdog):
    """Transformers SDPA, fp16 (bf16 only on nonfinite), left padding/LAST/L2."""
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    start=time.monotonic();dest.mkdir(parents=True,exist_ok=True)
    name,revision=MODELS[tier]
    tokenizer=AutoTokenizer.from_pretrained(name,revision=revision,padding_side='left')
    dtype=os.environ.get('VI51_DTYPE','fp16')
    assert dtype in ('fp16','bf16')
    model=AutoModel.from_pretrained(name,revision=revision,torch_dtype=getattr(torch,'float16' if dtype=='fp16' else 'bfloat16'),
        attn_implementation='sdpa').to('cuda').eval()
    budget=int(os.environ.get('VI51_TOKEN_BUDGET','65536' if tier=='a' else '16384'))
    pc='LAST/L2; left padding; transformers SDPA '+dtype
    def embed(rows):
        nonlocal dtype,pc
        e=np.empty((len(rows),model.config.hidden_size),dtype=np.float32)
        for indices in token_batches(rows,budget):
            assert time.monotonic()-start<watchdog-5
            batch=tokenizer.pad([dict(input_ids=rows[i]['input_ids']) for i in indices],padding=True,return_tensors='pt').to('cuda')
            assert bool(torch.all(batch['attention_mask'][:,-1]==1))
            with torch.inference_mode():
                last=model(**batch).last_hidden_state[:,-1].float()
                vectors=torch.nn.functional.normalize(last,p=2,dim=1)
            e[indices]=vectors.cpu().numpy()
            del batch,last,vectors
        if not np.isfinite(e).all():
            if dtype=='fp16':
                print('NONFINITE fp16: permitted fallback bf16',flush=True)
                dtype='bf16';pc='LAST/L2; left padding; transformers SDPA bf16'
                model.to(dtype=torch.bfloat16)
                return embed(rows)
            raise FloatingPointError('nonfinite embeddings dtype='+dtype)
        assert np.max(np.abs(np.linalg.norm(e,axis=1)-1))<2e-4
        return e
    if mode=='bench':
        # Operational sample: first 2,000 real rows in prepared part000, no input rebuild.
        rows=list(iterz(source/('docs51.part000.jsonl.zst' if tier=='a' else 'docs51b.part000.jsonl.zst')))[:2000]
        # Warm-up excluded from throughput; charged in total function wall.
        embed(rows[:4]); t=time.monotonic();e=embed(rows);seconds=time.monotonic()-t
        repeat=embed(rows[:50]);cos=np.sum(e[:50]*repeat,axis=1)
        np.savez(dest/'bench_embeddings.npz',ids=np.asarray([r['id'] for r in rows]),emb=e)
        report=dict(model=name,revision=revision,release=RELEASE,documents=len(rows),
            tokens=sum(len(r['input_ids']) for r in rows),seconds=seconds,
            tokens_per_second=sum(len(r['input_ids']) for r in rows)/seconds,
            repeat_cosine_min=float(cos.min()),repeat_cosine_mean=float(cos.mean()),repeat_cosine_max=float(cos.max()),pooler=str(pc),dtype=dtype,
            backend='transformers+sdpa',torch=torch.__version__,transformers=transformers.__version__,token_budget=budget,
            function_seconds=time.monotonic()-start)
        if tier=='a':
            queries=read(source/'queries.json')[:5];qe=embed(queries);sim=qe@e.T
            report['sanity_top10']=[dict(query_id=q['id'],doc_ids=[rows[j]['id'] for j in np.argsort(-sim[i])[:10]],
                cosines=sorted(map(float,sim[i]),reverse=True)[:10],std=float(np.std(sim[i]))) for i,q in enumerate(queries)]
            report['sanity_pass']=bool(np.isfinite(sim).all() and np.all(np.std(sim,axis=1)>1e-6))
        write(dest/'bench_report.json',report)
        print('BENCH_REPORT '+json.dumps(report),flush=True)
        assert repeat_pass(report['repeat_cosine_min'],report['repeat_cosine_mean']),report
        assert report.get('sanity_pass',True),report
        return report
    queries=read(source/('queries.json' if tier=='a' else 'queries_b.json'));qe=embed(queries)
    np.savez(dest/'query_embeddings.npz',ids=np.asarray([r['id'] for r in queries]),emb=qe)
    pool=read(source/'pool.json'); reverse=defaultdict(list)
    for qi,q in enumerate(queries):
        for d in pool[str(q['id'])]: reverse[d].append(qi)
    scores={str(q['id']):{} for q in queries};reports=[];seen=set()
    for part in parts:
        assert time.monotonic()-start<watchdog-10
        path=source/part
        # Preparation independently tokenized both tiers and proved SHA-identical
        # all 29 compressed parts. Reuse exact bytes instead of duplicate uploads.
        if not path.exists() and tier=='b':path=source/part.replace('docs51b.','docs51.')
        rows=list(iterz(path));assert all(r['id'] not in seen for r in rows)
        t=time.monotonic();e=embed(rows)
        # Persist compact embeddings separately; cosine is still computed from float32.
        np.savez(dest/(part+'.emb.npz'),ids=np.asarray([r['id'] for r in rows]),emb=e.astype(np.float16))
        for i,r in enumerate(rows):
            d=r['id'];seen.add(d)
            for qi in reverse[d]: scores[str(queries[qi]['id'])][str(d)]=float(e[i]@qe[qi])
        reports.append(dict(part=part,documents=len(rows),tokens=sum(len(r['input_ids']) for r in rows),seconds=time.monotonic()-t))
        write(dest/'partial_report.json',dict(parts=reports,function_seconds=time.monotonic()-start))
        print('PART_DONE '+json.dumps(reports[-1]),flush=True)
    from vi37 import writez
    writez(dest/'cosines.json.zst',scores)
    report=dict(model=name,revision=revision,release=RELEASE,parts=reports,documents=len(seen),
        pairs=sum(map(len,scores.values())),function_seconds=time.monotonic()-start,
        backend='transformers+sdpa',dtype=dtype,torch=torch.__version__,transformers=transformers.__version__,token_budget=budget,
        files={p.name:fingerprint(p) for p in dest.iterdir() if p.is_file()})
    write(dest/'bulk_report.json',report);return report


def build(out, tier, shared):
    import dwmax_diag as dw
    from vi41c import load_zip,digest,indexed
    from vi29 import assert_clean,load_corpus_ids
    from reorder_zip import write_reordered_zip
    from scipy.stats import spearmanr
    start=time.monotonic();tag='vi51'+tier
    zp=out/f'submission_{tag}.zip';jp=out/f'submission_{tag}.json'
    assert not zp.exists() and not (out/'build_report.json').exists(), 'ONE build only'
    ctx=dw.read(shared/'fusion_context.json.zst');prep=read(shared/'prepare_report.json')
    scores={q:{} for q in ctx['data']['pool']};seen=set();ndoc=0
    scorefiles=sorted((out/'outputs').rglob('cosines.json.zst'))
    assert scorefiles
    for p in scorefiles:
        for q,ds in dw.read(p).items():
            assert not set(scores[q])&set(ds)
            assert all(math.isfinite(v) for v in ds.values())
            scores[q].update(ds)
    for q,hits in ctx['data']['pool'].items():
        assert set(scores[q])=={str(h[0]) for h in hits},('incomplete pool',q)
    dw.write(shared/f'cosines_{tag}.json',scores)
    rankings=fused(ctx['data'],ctx['bge'],scores)
    base=sorted(load_zip(TEMP/'vi48/submission_vi48.zip'),key=lambda r:r['id'])
    assert len(base)==1200 and set(indexed(base))==set(range(1,1201))
    hashes_chunks={r['id']:digest(r['relevant_chunks']) for r in base}
    counts={r['id']:len(r['relevant_docs']) for r in base}
    rows=[];changes=[];protected=0
    for r in base:
        row,m=protect(r,rankings[str(r['id'])]);protected+=m
        a,b=set(row['relevant_docs']),set(r['relevant_docs'])
        changes.append(dict(id=r['id'],entering=sorted(a-b),leaving=sorted(b-a)))
        rows.append(row)
    del base;gc.collect()
    x=[];bg=[];fh=[];mb=[];mf=[];overlaps=[]
    for q,ds in scores.items():
        dense={str(h[0]):float(h[2]) for h in ctx['data']['dense'][q]}
        old={str(h[0]):float(h[2]) for h in ctx['rankings'][q]}
        common=[d for d in ds if d in dense]
        a=[ds[d] for d in common];b=[dense[d] for d in common];c=[old[d] for d in common]
        x.extend(a);bg.extend(b);fh.extend(c)
        for dest,other in ((mb,b),(mf,c)):
            rho=float(spearmanr(a,other).statistic)
            if math.isfinite(rho):dest.append(rho)
        qr=sorted(ctx['data']['pool'][q],key=lambda h:-ds[str(h[0])])
        overlaps.append(len({str(h[0]) for h in qr[:30]}&{str(h[0]) for h in ctx['data']['dense'][q][:30]}))
    allx=[];allf=[]
    for q,ds in scores.items():
        old={str(h[0]):h[2] for h in ctx['rankings'][q]}
        allx.extend(ds.values());allf.extend(old[d] for d in ds)
    offline=dict(changed_queries=sum(bool(r['entering']) for r in changes),
        entering=sum(len(r['entering']) for r in changes),leaving=sum(len(r['leaving']) for r in changes),
        entering_per_query=sum(len(r['entering']) for r in changes)/1200,
        spearman_q3e_dense_pooled=float(spearmanr(x,bg).statistic),spearman_q3e_dense_macro=sum(mb)/len(mb),
        spearman_q3e_fhyde_pooled=float(spearmanr(allx,allf).statistic),
        top30_dense_overlap_mean=sum(overlaps)/len(overlaps),correlation_dense_pairs=len(x),
        protected_docs=protected,per_query=changes)
    write(out/'offline.json',offline)
    source=TEMP/'vi45'
    bad=set(map(int,read(source/'bad_ids.json')))-set(map(int,read(source/'new_ids.json')))
    clean=assert_clean(rows,bad,load_corpus_ids(source/'links_corpus_vi_en_zh.parquet'))
    assert all(len(r['relevant_docs'])==len(set(r['relevant_docs'])) for r in rows)
    assert all(digest(r['relevant_chunks'])==hashes_chunks[r['id']] and len(r['relevant_docs'])==counts[r['id']] for r in rows)
    hashes={r['id']:digest(r) for r in rows}
    del ctx,scores,rankings,allx,allf;gc.collect()
    info=write_reordered_zip(rows,zp)  # exactly ONE compression / no tuning
    assert zp.stat().st_size<=99_000_000
    del rows;gc.collect()
    with zipfile.ZipFile(zp) as z:
        assert z.namelist()==['submission.json'] and z.infolist()[0].compress_type==zipfile.ZIP_LZMA
        assert z.testzip() is None
        with z.open('submission.json') as src,jp.open('xb') as dst:shutil.copyfileobj(src,dst)
    final=load_zip(zp)
    assert {r['id']:digest(r) for r in final}==hashes
    assert all(digest(r['relevant_chunks'])==hashes_chunks[r['id']] and len(r['relevant_docs'])==counts[r['id']] for r in final)
    del final;gc.collect()
    subprocess.run([sys.executable,str(Path(__file__).with_name('evaluate.py')),'validate',str(jp)],check=True)
    subprocess.run([sys.executable,str(Path(__file__).resolve()),'check-zip','--out',str(out),'--tier',tier],check=True)
    report=dict(**info,**fingerprint(zp),validate=0,check_zip=0,assert_clean=clean,crc=0,ids=1200,
        chunks_equal_vi48=1200,doc_counts_equal_vi48=1200,entry='submission.json',compression='LZMA',
        build_attempts=1,pairs=prep['pairs'],documents=prep['documents'],seconds=time.monotonic()-start)
    write(out/'build_report.json',report);print('BUILD_DONE '+json.dumps(report),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','gpu','build','check-zip'])
    ap.add_argument('--root',type=Path,default=Path('D:/r2ai'));ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--tier',choices=['a','b'],default='a');ap.add_argument('--mode',choices=['bench','bulk'])
    ap.add_argument('--source',type=Path);ap.add_argument('--shared',type=Path,default=TEMP/'vi51')
    ap.add_argument('--parts',nargs='*',default=[]);ap.add_argument('--watchdog',type=int,default=1200)
    a=ap.parse_args()
    if a.stage=='check-zip':
        from vi23 import check_zip
        tag='vi51'+a.tier
        raise SystemExit(check_zip(str(a.out/f'submission_{tag}.json'),str(a.out/f'submission_{tag}.zip'),str(a.out/'check_zip.json'),tag,max_zip=99_000_000))
    if a.stage=='gpu':return gpu(a.tier,a.mode,a.source,a.out,a.parts,a.watchdog)
    from dwmax_diag import owned_lock
    lock=a.root/'locks/heavy_job.lock' if a.stage=='prepare' else a.out/'build.lock'
    while lock.exists() and a.stage=='prepare':time.sleep(10)
    with owned_lock(lock,f'vi51 {a.stage} pid={os.getpid()}'):
        prepare(a.root,a.out) if a.stage=='prepare' else build(a.out,a.tier,a.shared)


if __name__=='__main__':main()
