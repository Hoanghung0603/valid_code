"""vi55 phase 1 only: immutable VI corpus, Qwen8B retrieval, no submission."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

MODEL = 'Qwen/Qwen3-Embedding-8B'
REVISION = '1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'
INSTRUCT = 'Given a Vietnamese medical question, retrieve web documents that answer the question'
RATE = 4.5232 / 3600


def query_text(q):
    return f'Instruct: {INSTRUCT}\nQuery:{q}'


def write(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, allow_nan=False, indent=2), encoding='utf-8')
    tmp.replace(path)


def read(path):
    path = Path(path)
    if path.suffix == '.zst':
        import zstandard as zstd
        with zstd.open(path, 'rt', encoding='utf-8') as f: return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def iterz(path):
    import zstandard as zstd
    with zstd.open(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def fingerprint(path):
    path = Path(path); h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def vi_corpus(rows):
    """Official concatenation: first VI block, then PubMed, then ZH; verify boundary."""
    from urllib.parse import urlsplit
    boundary = next(i for i, r in enumerate(rows) if urlsplit(r['url']).hostname == 'pubmed.ncbi.nlm.nih.gov')
    assert boundary == 740598 and len(rows) == 1220598
    assert all(urlsplit(r['url']).hostname == 'pubmed.ncbi.nlm.nih.gov' for r in rows[boundary:boundary+240000])
    ids = {int(r['id']) for r in rows[:boundary]}
    assert len(ids) == boundary
    return ids


def prepare(root, out):
    import numpy as np
    import pyarrow.parquet as pq
    import zstandard as zstd
    from transformers import AutoTokenizer
    from bigindex import BigIndex
    from boilerplate import BoilerplateFilter
    from export_docs import iter_all_texts
    from dwmax_diag import safe_index
    start = time.monotonic(); out.mkdir(parents=True, exist_ok=True)
    assert not (out/'prepare_report.json').exists(), 'Do not overwrite completed prepare'
    inp = out/'modal_input'; inp.mkdir(exist_ok=True)
    corpus = root/'data/sub/diag19/links_corpus_vi_en_zh.parquet'
    allowed = vi_corpus(pq.read_table(corpus).to_pylist())
    index = root/'data/index/vi36'; safe_index(index); idx = BigIndex(index)
    wanted = {int(d): i for i, d in enumerate(idx.ids) if int(d) in allowed}
    pool = read(root/'data/sub/vi51/modal_input/pool.json')
    historical = {int(d) for ds in pool.values() for d in ds}
    assert len(historical) == 223065
    old = historical & allowed
    print('SCOPE_AUDIT '+json.dumps(dict(old=len(historical),reusable_vi=len(old),wanted=len(wanted),historical_non_vi=sorted(historical-allowed))),flush=True)
    assert old <= wanted.keys()
    # Existing fp16 Qwen8B embeddings are audited separately and kept at 2048 tokens.
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, padding_side='left')
    queries = [dict(id=int(q['id']), input_ids=tok(query_text(q['query']), truncation=True, max_length=256)['input_ids'])
               for q in pq.read_table(root/'data/query.parquet').to_pylist()]
    assert len(queries) == 1200 and {q['id'] for q in queries} == set(range(1,1201))
    write(inp/'queries.json', queries); write(out/'pool.json', pool)
    np.save(out/'old_ids.npy', np.asarray(sorted(old), dtype=np.int64))
    bp = BoilerplateFilter.from_json((index/'boilerplate.json').read_text(encoding='utf-8'))
    seen = set(); nonempty = set(); empty = []; parts = {}; batch = []; block = []
    tokens1024 = tokens512 = 0; bench = []
    def save_block():
        nonlocal block
        if not block: return
        p = inp/f'docs55.part{len(parts):03d}.jsonl.zst'
        with zstd.open(p, 'wt', encoding='utf-8') as f:
            for r in block: f.write(json.dumps(r, ensure_ascii=False)+'\n')
        parts[p.name] = dict(**fingerprint(p), docs=len(block), tokens1024=sum(len(r['input_ids']) for r in block),
                            tokens512=sum(min(512,len(r['input_ids'])) for r in block))
        print('PART '+p.name+' '+json.dumps(parts[p.name]), flush=True); block = []
    def flush():
        nonlocal batch, tokens1024, tokens512
        if not batch: return
        encoded = tok([r[1] for r in batch], truncation=True, max_length=1024)['input_ids']
        for (d, _), ids in zip(batch, encoded):
            assert ids
            row = dict(id=d, input_ids=ids); block.append(row)
            tokens1024 += len(ids); tokens512 += min(512,len(ids))
            if len(bench)<2000: bench.append(row)
            if len(block)>=10000: save_block()
        batch = []
    # Read exactly the same immutable index/shards and cleaning as docs54, no char cut.
    for d, i, text in iter_all_texts(idx, wanted, bp, 2**31-1):
        assert d not in seen; seen.add(d)
        if not text.strip(): empty.append(d); continue
        nonempty.add(d)
        if d in old: continue
        batch.append((d,text))
        if len(batch)>=128: flush()
    flush(); save_block()
    assert old <= nonempty
    np.save(out/'all_ids.npy', np.asarray(sorted(nonempty),dtype=np.int64))
    with zstd.open(inp/'bench.jsonl.zst','wt',encoding='utf-8') as f:
        for r in bench: f.write(json.dumps(r)+'\n')
    dense_candidates = [root/'data/sub/vi36/dense_full_top400.json', root/'data/sub/vi30/dense_full_top400.json',
                        root/'data/sub/diag30/dense_full_top400.json']
    # Discover filename only under sub, never write old artifacts.
    dense = next((p for p in dense_candidates if p.exists()), None)
    if dense is None:
        candidates = list((root/'data/sub').rglob('dense_full_top400.json'))
        if candidates: dense = max(candidates,key=lambda p:p.stat().st_mtime)
    if dense:
        hits = read(dense); write(out/'dense_top200.json', {q:rows[:200] for q,rows in hits.items()})
    report = dict(corpus_vi=len(allowed), index_vi=len(wanted), with_text=len(nonempty), without_text=len(allowed-nonempty),
                  index_unreadable=sorted(wanted.keys()-seen), empty_text=empty, reused_documents=len(old),
                  historical_pool_documents=len(historical),historical_pool_non_vi=sorted(historical-allowed),
                  new_documents=len(nonempty-old), tokens1024=tokens1024, tokens512=tokens512,
                  query_tokens=sum(len(q['input_ids']) for q in queries), parts=parts,
                  source_index=str(index), corpus=fingerprint(corpus), dense_source=str(dense) if dense else None,
                  cleaning='export_docs.iter_all_texts/BoilerplateFilter.clean/make_body(title,clean,2**31-1)',
                  model=MODEL,revision=REVISION,seconds=time.monotonic()-start)
    write(out/'prepare_report.json',report); print('PREPARE_DONE '+json.dumps(report),flush=True)


def token_batches(rows, budget=16384):
    batch=[]
    for i in sorted(range(len(rows)),key=lambda i:(len(rows[i]['input_ids']),i)):
        if batch and len(rows[i]['input_ids'])*(len(batch)+1)>budget:
            yield batch; batch=[]
        batch.append(i)
    if batch: yield batch


def repeat_pass(minimum, mean):
    return minimum >= .999 and mean >= .9999


def forecast(tokens, throughput, jobs=8, overhead=40):
    seconds = 1.3*tokens/throughput + jobs*overhead
    return dict(tokens=tokens, throughput=throughput, margin=1.3, gpu_seconds=seconds, usd=seconds*RATE)


def audit_embedding(path, expected_ids, dimension=4096, dtype='float16'):
    import numpy as np
    expected_ids = np.asarray(expected_ids, dtype=np.int64)
    assert len(set(map(int, expected_ids))) == len(expected_ids)
    with np.load(path, allow_pickle=False) as z:
        ids, emb = z['ids'], z['emb']
        assert np.array_equal(ids, expected_ids), 'Incomplete or wrong part IDs/order'
        assert emb.shape == (len(ids), dimension) and emb.dtype == np.dtype(dtype)
        assert np.isfinite(emb).all()
        error = float(np.max(abs(np.linalg.norm(emb.astype(np.float32), axis=1)-1)))
        assert error < .001, error
    return dict(**fingerprint(path), documents=len(expected_ids), shape=[len(expected_ids), dimension],
                dtype=dtype, l2_max_error=error, ids=list(map(int, expected_ids)))


def committed_part(source, dest, part, dimension=4096):
    """A filename or a partial report is never evidence of completion."""
    import zipfile
    target = dest/(part+'.emb.npz'); meta = dest/(part+'.complete.json')
    if not target.exists() or not meta.exists(): return None
    try:
        report = read(meta)
        assert report['input'] == fingerprint(source/part)
        expected = [r['id'] for r in iterz(source/part)]
        audit = audit_embedding(target, expected, dimension)
        assert audit == report['output']
        return report
    except (AssertionError, ValueError, OSError, KeyError, EOFError, zipfile.BadZipFile):
        return None


def process_part(source, dest, part, embed, doc_len=1024, dimension=4096, commit=lambda: None):
    import numpy as np
    dest.mkdir(parents=True, exist_ok=True)
    existing = committed_part(source, dest, part, dimension)
    if existing is not None: return dict(existing, skipped=True)
    attempt_path = dest/(part+'.attempts.json')
    attempts = read(attempt_path) if attempt_path.exists() else []
    assert len(attempts) < 2, f'Part preempted/failed twice: {part}; STOP'
    t = time.monotonic()
    attempts.append(dict(start=time.time(), status='started'))
    write(attempt_path, attempts); commit()
    try:
        rows = [dict(id=r['id'], input_ids=r['input_ids'][:doc_len]) for r in iterz(source/part)]
        e = embed(rows)
        target = dest/(part+'.emb.npz')
        temporary = target.with_suffix('.pending.npz')
        np.savez(temporary, ids=np.asarray([r['id'] for r in rows], dtype=np.int64), emb=e.astype(np.float16))
        audit = audit_embedding(temporary, [r['id'] for r in rows], dimension)
        temporary.replace(target)
        report = dict(part=part, documents=len(rows), tokens=sum(len(r['input_ids']) for r in rows),
                      seconds=time.monotonic()-t, input=fingerprint(source/part), output=audit,
                      attempt=len(attempts), completed=time.time())
        write(dest/(part+'.complete.json'), report)
        attempts[-1].update(status='done', seconds=time.monotonic()-t)
        write(attempt_path, attempts); commit()
        return report
    except BaseException:
        attempts[-1].update(status='interrupted', seconds=time.monotonic()-t)
        write(attempt_path, attempts); commit()
        raise


def gpu(mode, source, dest, parts, watchdog, doc_len, commit=lambda: None):
    import numpy as np
    import torch
    import transformers
    from transformers import AutoModel, AutoTokenizer
    start=time.monotonic(); dest.mkdir(parents=True,exist_ok=True)
    assert read('/weights/complete.json')['revision']==REVISION
    tok=AutoTokenizer.from_pretrained('/weights/model',padding_side='left',local_files_only=True)
    model=AutoModel.from_pretrained('/weights/model',torch_dtype=torch.float16,attn_implementation='sdpa',local_files_only=True).cuda().eval()
    def embed(rows):
        result=np.empty((len(rows),model.config.hidden_size),np.float32)
        for indices in token_batches(rows):
            assert time.monotonic()-start<watchdog-5
            b=tok.pad([dict(input_ids=rows[i]['input_ids']) for i in indices],padding=True,return_tensors='pt').to('cuda')
            assert bool(torch.all(b['attention_mask'][:,-1]==1))
            with torch.inference_mode():
                last=model(**b).last_hidden_state[:,-1].float()
                vectors=torch.nn.functional.normalize(last,p=2,dim=1)
            result[indices]=vectors.cpu().numpy()
        assert np.isfinite(result).all() and np.max(abs(np.linalg.norm(result,axis=1)-1))<2e-4
        return result
    common=dict(model=MODEL,revision=REVISION,dtype='fp16',backend='SDPA',pooling='LAST/L2',padding='left',
                doc_len=doc_len,query_len=256,torch=str(torch.__version__),transformers=transformers.__version__)
    if mode=='gate':
        rows=list(iterz(source/parts[0]))[:50]
        e=embed(rows); again=embed(rows); cos=np.sum(e*again,axis=1)
        report=dict(**common,documents=50,repeat_cosine_min=float(cos.min()),repeat_cosine_mean=float(cos.mean()),
                    repeat_cosine_max=float(cos.max()),function_seconds=time.monotonic()-start)
        write(dest/'gate_report.json',report); commit(); print('REPEAT_REPORT '+json.dumps(report),flush=True)
        assert repeat_pass(report['repeat_cosine_min'],report['repeat_cosine_mean']),report
        return report
    if mode=='bench':
        rows=list(iterz(source/'bench.jsonl.zst'))
        rows=[dict(id=r['id'],input_ids=r['input_ids'][:doc_len]) for r in rows]
        embed(rows[:4]); t=time.monotonic(); e=embed(rows); seconds=time.monotonic()-t
        again=embed(rows[:50]); cos=np.sum(e[:50]*again,axis=1)
        report=dict(**common,documents=len(rows),tokens=sum(len(r['input_ids']) for r in rows),seconds=seconds,
                    tokens_per_second=sum(len(r['input_ids']) for r in rows)/seconds,
                    repeat_cosine_min=float(cos.min()),repeat_cosine_mean=float(cos.mean()),repeat_cosine_max=float(cos.max()))
        write(dest/'bench_report.json',report); print('REPEAT_REPORT '+json.dumps(report),flush=True)
        assert repeat_pass(report['repeat_cosine_min'],report['repeat_cosine_mean']),report
        qe=embed(read(source/'queries.json')[:5]); sim=qe@e.T
        report.update(sanity_pass=bool(np.isfinite(sim).all() and np.all(np.std(sim,axis=1)>1e-6)),
                      sanity_std=list(map(float,np.std(sim,axis=1))),function_seconds=time.monotonic()-start)
        write(dest/'bench_report.json',report); assert report['sanity_pass']; return report
    if mode=='query':
        queries=read(source/'queries.json')
        assert len(queries)==1200 and {q['id'] for q in queries}==set(range(1,1201))
        qe=embed(queries)
        np.savez(dest/'query_embeddings.npz',ids=np.asarray([q['id'] for q in queries]),emb=qe)
        audit=audit_embedding(dest/'query_embeddings.npz',[q['id'] for q in queries],model.config.hidden_size,'float32')
        report=dict(**common,output=audit,function_seconds=time.monotonic()-start)
        write(dest/'query_report.json',report);commit();return report
    reports=[]
    for part in parts:
        reports.append(process_part(source,dest,part,embed,doc_len,model.config.hidden_size,commit))
        write(dest/'partial_report.json',dict(parts=reports,function_seconds=time.monotonic()-start))
        commit()
        print('PART_DONE '+json.dumps(reports[-1]),flush=True)
    report=dict(**common,parts=reports,documents=sum(r['documents'] for r in reports),function_seconds=time.monotonic()-start,
                files={p.name:fingerprint(p) for p in dest.iterdir() if p.is_file()})
    write(dest/'bulk_report.json',report); return report


def merge_topk(scores, ids, new_scores, new_ids, k):
    import numpy as np
    ss=np.concatenate([scores,new_scores],axis=1); ii=np.concatenate([ids,new_ids],axis=1)
    # Stable deterministic tie-break by doc id; chunks are disjoint.
    order=np.lexsort((ii,-ss),axis=1)[:,:k]
    return np.take_along_axis(ss,order,1),np.take_along_axis(ii,order,1)


def metrics(top, pool, dense, doclists, exact_ranks=None):
    import numpy as np
    assert top.keys()==pool.keys()==dense.keys()==doclists.keys()
    def dist(values):
        return dict(mean=float(np.mean(values)),p50=float(np.percentile(values,50)),p90=float(np.percentile(values,90)))
    result={}; all_ranks=[]; missing=0
    for k in (100,200):
        new=[]; global_new=[]; overlap=[]; coverage=[]; total=found=0
        global_pool={int(d) for ds in pool.values() for d in ds}
        for q, rows in top.items():
            ids={int(r[0]) for r in rows[:k]}; old=set(map(int,pool[q])); ds=set(map(int,doclists[q]))
            new.append(len(ids-old));global_new.append(len(ids-global_pool)); overlap.append(len(ids&{int(h[0]) for h in dense[q][:200]}))
            n=len(ids&ds); found+=n; total+=len(ds); coverage.append(n/len(ds) if ds else 0)
        result[str(k)]=dict(outside_query_pool=dist(new),outside_global_pool=dist(global_new),overlap_bge_top200=dist(overlap),
                            overlap_fraction_k=float(np.mean(overlap))/k,
                            vi53_coverage_micro=found/total,vi53_coverage_macro=float(np.mean(coverage)),vi53_found=found,vi53_total=total)
    new30=[]; global_pool={int(d) for ds in pool.values() for d in ds}; global_new30=[]
    for q, rows in top.items():
        ids=[int(r[0]) for r in rows[:30]]; old=set(map(int,pool[q]))
        new30.append(sum(d not in old for d in ids)); global_new30.append(sum(d not in global_pool for d in ids))
        rank={int(r[0]):i+1 for i,r in enumerate(rows)}
        for d in doclists[q]:
            v=exact_ranks.get(q,{}).get(str(d)) if exact_ranks is not None else rank.get(int(d))
            if v is None: missing+=1
            else: all_ranks.append(v)
    result['top30_proxy']=dict(new_fraction=float(np.mean(new30))/30,old_fraction=1-float(np.mean(new30))/30,
                               new_to_old_ratio=sum(new30)/(30*len(new30)-sum(new30)) if sum(new30)<30*len(new30) else None,
                               new_count=dist(new30),outside_global_pool_fraction=float(np.mean(global_new30))/30)
    result['vi53_rank']=dict(**dist(all_ranks),ranked_pairs=len(all_ranks),exact=exact_ranks is not None,missing_or_beyond1000=missing)
    cosine={key:[] for key in ('new_to_query_pool','old_in_query_pool','new_to_global_pool','old_in_global_pool')}
    for q,rows in top.items():
        old=set(map(int,pool[q]))
        for d,s in rows[:30]:
            cosine['old_in_query_pool' if int(d) in old else 'new_to_query_pool'].append(s)
            cosine['old_in_global_pool' if int(d) in global_pool else 'new_to_global_pool'].append(s)
    result['top30_cosine']={key:dict(n=len(values),**dist(values),min=float(min(values)),max=float(max(values)))
                            if values else dict(n=0) for key,values in cosine.items()}
    result['definitions']=dict(outside_query_pool='doc absent from that query vi41c candidate pool',
                               outside_global_pool='doc absent from union of all vi41c candidate pools',
                               vi53_rank_denominator='VI documents with text; non-VI not ranked',
                               coverage_denominator='all vi53 query-document predictions including non-VI')
    return result


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['prepare','gpu'])
    ap.add_argument('--root',type=Path,default=Path('D:/r2ai')); ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--source',type=Path); ap.add_argument('--mode',choices=['bench','bulk','gate','query'])
    ap.add_argument('--parts',nargs='*',default=[]); ap.add_argument('--watchdog',type=int,default=240)
    ap.add_argument('--doc-len',type=int,choices=[512,1024],default=1024)
    a=ap.parse_args()
    if a.stage=='gpu': return gpu(a.mode,a.source,a.out,a.parts,a.watchdog,a.doc_len)
    from dwmax_diag import owned_lock
    lock=a.root/'locks/heavy_job.lock'
    while lock.exists(): time.sleep(10)
    with owned_lock(lock,f'vi55 prepare pid={os.getpid()}'): prepare(a.root,a.out)


if __name__=='__main__': main()
