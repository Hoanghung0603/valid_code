"""Budgeted dense CLS; same weights/pooling, no vector download.

modal run deploy/modal_dchunk.py --stage bench
Production resume: modal deploy deploy/modal_dchunk.py, then dchunk_service.py.
Deployed Function.spawn + persisted call ID only; embed/search are separate calls.
Function watchdogs and hard timeouts; zero retries, minimum 2s GPU scaledown.
"""
import modal

MODEL = 'BAAI/bge-m3'
RATE_UPPER = (1.95 + 2*.0473 + 16*.008)/3600
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.6.0', 'transformers==4.51.3', 'numpy==1.26.4',
                      'huggingface_hub==0.30.2', 'safetensors==0.5.3', 'sentencepiece==0.2.0')
         .env({'HF_HOME':'/hf', 'HF_HUB_CACHE':'/hf', 'TOKENIZERS_PARALLELISM':'false'})
         .run_commands("python -c \"from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', allow_patterns=['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model'])\""))
vol = modal.Volume.from_name('r2ai-runner', create_if_missing=False)
app = modal.App('r2ai-dchunk-svc', image=image)


@app.function(cpu=2, memory=2048, timeout=120, retries=0, volumes={'/vol':vol})
def audit_volume():
    import hashlib
    import json
    import time
    from pathlib import Path
    start = time.time()
    src = Path('/vol/in/dchunk')
    manifest = json.loads((src/'manifest.json').read_text())
    for part in manifest['files']:
        p = src/part['name']
        assert p.stat().st_size==part['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==part['sha256'], part['name']
    rep = dict(files=len(manifest['files']),bytes=manifest['bytes'],sha_errors=0,
               manifest_sha256=hashlib.sha256((src/'manifest.json').read_bytes()).hexdigest(),
               small_sha256={n:hashlib.sha256((src/n).read_bytes()).hexdigest() for n in ('bench.npz','queries.json')},
               elapsed_seconds=time.time()-start)
    dest=Path('/vol/out/dchunk')
    dest.mkdir(parents=True,exist_ok=True)
    (dest/'volume_audit.json').write_text(json.dumps(rep,indent=2))
    vol.commit()
    return rep


@app.function(gpu='L40S', cpu=2, memory=16384, timeout=90, scaledown_window=2, retries=0, volumes={'/vol':vol})
def bench():
    return run_job('bench', 75, 'L40S', 1.95)


@app.function(gpu='H100', cpu=2, memory=16384, timeout=90, scaledown_window=2, retries=0, volumes={'/vol':vol})
def bench_h100():
    return run_job('bench', 75, 'H100', 4.54)


@app.function(gpu='L40S', cpu=2, memory=16384, timeout=6500, scaledown_window=2, retries=0, volumes={'/vol':vol})
def bulk(seconds_budget: int, scope: str='full'):
    assert 60 <= seconds_budget <= 6400
    return run_job('bulk', seconds_budget, scope=scope)


@app.function(gpu='H100', cpu=2, memory=16384, timeout=900, scaledown_window=2, retries=0, volumes={'/vol':vol})
def bulk_h100(seconds_budget: int, scope: str='full', resume: bool=False, phase: str='embed'):
    assert 60 <= seconds_budget <= 850 and phase == 'embed' and resume
    return run_job('bulk', seconds_budget, 'H100', 4.54, scope, resume, phase)


@app.function(gpu='H100', cpu=2, memory=16384, timeout=300, scaledown_window=2, retries=0, volumes={'/vol':vol})
def search_h100(seconds_budget: int):
    assert 60 <= seconds_budget <= 260
    return run_job('bulk', seconds_budget, 'H100', 4.54, 'full', True, 'search')


def validate_vectors(vectors, chunk_ids, expected_ids):
    import numpy as np
    assert vectors.shape == (len(expected_ids), 1024)
    assert vectors.dtype == np.float16 and chunk_ids.dtype == np.int64
    assert np.array_equal(chunk_ids, expected_ids)
    assert np.isfinite(vectors).all()
    norms = np.linalg.norm(vectors.astype(np.float32), axis=1)
    assert norms.min() > .999 and norms.max() < 1.001
    return dict(norm_min=float(norms.min()), norm_max=float(norms.max()))


@app.function(cpu=2, memory=2048, timeout=180, retries=0, volumes={'/vol':vol})
def audit_partial():
    import hashlib
    import json
    import time
    from pathlib import Path
    import numpy as np
    t0 = time.time()
    src, dest = Path('/vol/in/dchunk'), Path('/vol/out/dchunk')
    manifest = json.loads((src/'manifest.json').read_text())
    known = {p['name'] for p in manifest['files']}
    assert {p.name for p in (dest/'vectors').glob('*.npz')} <= known
    bench_ids = np.load(src/'bench.npz')['chunk_id']
    good, bad, missing, samples, vectors = [], [], [], [], []
    offset = 0
    for part in manifest['files']:
        p = dest/'vectors'/part['name']
        if not p.exists():
            missing.append(part['name'])
        else:
            try:
                with np.load(p) as z:
                    e, c = z['vectors'], z['chunk_ids']
                stats = validate_vectors(e, c, np.arange(offset,offset+part['chunks'],dtype=np.int64))
                good.append(dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                                 chunks=len(c),tokens=part['tokens'],dtype=str(e.dtype),**stats))
                ix = np.flatnonzero(np.isin(c,bench_ids))[:16]
                samples.extend(c[ix].tolist())
                vectors.extend(e[ix])
            except (AssertionError,ValueError,KeyError,OSError) as exc:
                bad.append(dict(name=p.name,error=repr(exc)))
        offset += part['chunks']
    assert len(samples) >= 16, 'Need persisted benchmark-overlap vectors for cosine sanity'
    np.savez(dest/'partial_samples.npz',chunk_ids=np.asarray(samples,dtype=np.int64),vectors=np.asarray(vectors))
    plan = dict(manifest_sha256=hashlib.sha256((src/'manifest.json').read_bytes()).hexdigest(),
                good=good,bad=bad,missing=missing,run_parts=missing+[p['name'] for p in bad],
                sample_count=len(samples),elapsed_seconds=time.time()-t0)
    (dest/'resume_plan.json').write_text(json.dumps(plan,indent=2))
    vol.commit()
    return plan


def select_flat(ids, ptr, chunk_ids, selected):
    import numpy as np
    ix = np.flatnonzero(np.isin(chunk_ids,selected,assume_unique=True))
    lens = np.diff(ptr)[ix]
    flat = np.concatenate([ids[ptr[i]:ptr[i+1]] for i in ix]) if len(ix) else ids[:0]
    return flat, np.concatenate(([0],np.cumsum(lens))), chunk_ids[ix]


def run_job(stage, seconds_budget, gpu_kind='L40S', gpu_hourly=1.95, scope='full', resume=False, phase='all'):
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    t0 = time.time()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = True
    src, dest = Path('/vol/in/dchunk'), Path('/vol/out/dchunk')
    dest.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((src / 'manifest.json').read_text())
    plan = None
    if resume:
        import hashlib
        assert stage=='bulk' and scope=='full'
        plan = json.loads((dest/'resume_plan.json').read_text())
        budget = json.loads((dest/('search_budget.json' if phase=='search' else 'resume_budget.json')).read_text())
        assert plan['manifest_sha256']==hashlib.sha256((src/'manifest.json').read_bytes()).hexdigest()
        assert budget['plan_sha256']==hashlib.sha256((dest/'resume_plan.json').read_bytes()).hexdigest()
        assert budget['prior_upper_usd']+seconds_budget*(gpu_hourly+2*.0473+16*.008)/3600+.15 < 5
        assert budget['prior_upper_usd']+budget['predicted_remaining_usd'] < 5
        assert budget['phase']==phase and budget['seconds_budget']==seconds_budget
        if phase=='search':
            assert (dest/'embedding_complete.json').exists()
            if (dest/'bulk_report.json').exists() and (dest/'top1000.npz').exists():
                previous = json.loads((dest/'bulk_report.json').read_text())
                assert previous['top1000_sha256']==hashlib.sha256((dest/'top1000.npz').read_bytes()).hexdigest()
                return previous
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16,
                                      attn_implementation='sdpa').cuda().eval()
    model.eval()
    loaded = time.time()-t0
    report = dict(stage=stage, phase=phase, loaded_seconds=loaded, model=MODEL, dtype='fp16',
                  pooling='CLS; normalize fp32; store fp16', query_max_length=256,
                  content_tokens=512, special_tokens=2, gpu=torch.cuda.get_device_name(), gpu_kind=gpu_kind,
                  hourly_upper=gpu_hourly+2*.0473+16*.008)
    rate = report['hourly_upper']/3600
    def guard():
        if time.time()-t0 > seconds_budget:
            raise TimeoutError('DCHUNK BUDGET WATCHDOG')

    @torch.inference_mode()
    def encode_flat(ids, ptr, batch=128):
        n = len(ptr)-1
        lens = np.diff(ptr)
        order = np.argsort(lens, kind='stable')
        result = np.empty((n, 1024), dtype=np.float16)
        for j in range(0,n,batch):
            guard()
            ix = order[j:j+batch]
            m = int(lens[ix].max())
            x = np.full((len(ix),m), tok.pad_token_id, dtype=np.int64)
            mask = np.zeros_like(x)
            for k,i in enumerate(ix):
                x[k,:lens[i]] = ids[ptr[i]:ptr[i+1]]
                mask[k,:lens[i]] = 1
            h = model(input_ids=torch.from_numpy(x).cuda(), attention_mask=torch.from_numpy(mask).cuda()).last_hidden_state[:,0]
            h = torch.nn.functional.normalize(h.float(),dim=-1)
            assert bool(torch.isfinite(h).all())
            result[ix] = h.half().cpu().numpy()
        return result

    def flat_text(texts, maxlen):
        seqs = tok(texts, truncation=True, max_length=maxlen)['input_ids']
        ptr = np.concatenate(([0],np.cumsum([len(x) for x in seqs])))
        return np.concatenate(seqs), ptr

    if stage == 'bench':
        z = np.load(src / 'bench.npz')
        ids, ptr = z['ids'], z['ptr']
        # Warmup measured separately; same weights, no inference instruction.
        encode_flat(ids[:ptr[128]],ptr[:129])
        torch.cuda.synchronize()
        s = time.time()
        e = encode_flat(ids,ptr)
        torch.cuda.synchronize()
        elapsed = time.time()-s
        # Recompute 16 sequences through the unmodified historical eager path as sanity.
        expected = []
        with torch.inference_mode():
            for i in range(16):
                x = torch.tensor(ids[ptr[i]:ptr[i+1]],device='cuda',dtype=torch.long)[None]
                h = model(input_ids=x,attention_mask=torch.ones_like(x),output_attentions=True).last_hidden_state[:,0]
                expected.append(torch.nn.functional.normalize(h.float(),dim=-1).half().cpu().numpy()[0])
        cos = np.sum(e[:16].astype('float32')*np.asarray(expected).astype('float32'),axis=1)
        assert float(cos.min()) > .999
        norms = np.linalg.norm(e.astype('float32'),axis=1)
        assert float(norms.min()) > .999 and float(norms.max()) < 1.001
        # Hold out 30% overhead, entire startup, query+full corpus search reserve 120s.
        sec_bulk = (manifest['embedded_tokens_with_overlap_specials']/(len(ids)/elapsed))*1.3 + loaded + 120
        report.update(chunks=len(e), tokens=int(len(ids)), inference_seconds=elapsed,
                      chunks_per_second=len(e)/elapsed, tokens_per_second=len(ids)/elapsed,
                      sanity_min_cosine=float(cos.min()), predicted_bulk_seconds=sec_bulk,
                      predicted_bulk_usd=sec_bulk*rate, total_seconds=time.time()-t0,
                      max_memory_gib=torch.cuda.max_memory_allocated()/2**30)
        # Avoid silently extrapolating a changed sample or input manifest.
        import hashlib
        report['manifest_sha256'] = hashlib.sha256((src/'manifest.json').read_bytes()).hexdigest()
        assert len(ids)==manifest['bench_tokens'] and len(e)==manifest['bench_chunks']
        report['bench_usd_upper'] = report['total_seconds']*rate
        (dest / ('bench_'+gpu_kind+'.json')).write_text(json.dumps(report,indent=2))
        vol.commit()
        print('BENCH',json.dumps(report),flush=True)
        return report

    assert scope in ('full','expanded_pool')
    b = json.loads((dest / ('selected_report.json' if scope=='expanded_pool' else 'bench_report.json')).read_text())
    assert b['gpu_kind']==gpu_kind, 'Bulk must use the benchmark-selected GPU'
    assert b['predicted_bulk_usd'] <= 4, 'Bulk prediction >4 USD: STOP'
    assert seconds_budget*rate <= 4, 'Bulk upper limit >4 USD'
    if plan is not None:
        report.update(resume=True,run_parts=plan['run_parts'],reused_parts=[x['name'] for x in plan['good']],
                      prior_upper_usd=budget['prior_upper_usd'],predicted_remaining_usd=budget['predicted_remaining_usd'])
    if plan is not None and phase!='search':
        sample = np.load(dest/'partial_samples.npz')
        bz = np.load(src/'bench.npz')
        si,sp,sc = select_flat(bz['ids'],bz['ptr'],bz['chunk_id'],sample['chunk_ids'])
        se = encode_flat(si,sp)
        old = {int(c):v for c,v in zip(sample['chunk_ids'],sample['vectors'])}
        cosine = np.sum(se.astype('float32')*np.asarray([old[int(c)] for c in sc]).astype('float32'),axis=1)
        assert len(cosine)>=16 and cosine.min()>.999, 'Persisted vectors differ from benchmark-overlap recomputation'
        report['partial_bench_min_cosine']=float(cosine.min())
        report['partial_bench_compared']=len(cosine)
        (dest/'partial_cosine_report.json').write_text(json.dumps(report,indent=2))
        vol.commit()
    selected = None
    expected_chunks, expected_tokens = manifest['chunks'], manifest['embedded_tokens_with_overlap_specials']
    documents = manifest['documents']
    if scope=='expanded_pool':
        import hashlib
        sm = json.loads((src/'pool_scope.json').read_text())
        assert hashlib.sha256((src/'pool_scope.npz').read_bytes()).hexdigest()==sm['sha256']
        selected = np.load(src/'pool_scope.npz')['chunk_ids']
        assert len(np.unique(selected))==len(selected)==sm['chunks'] and np.all(np.diff(selected)>0)
        assert b['scope']==scope and b['full_predicted_bulk_usd']>4 and b['predicted_bulk_usd']<=3
        documents, expected_chunks, expected_tokens = sm['documents'],sm['chunks'],sm['tokens']
    report.update(scope=scope,scope_note=('Dense-chunk trên pool mở rộng, KHÔNG phải toàn kho' if selected is not None else 'Dense-chunk toàn kho VI'),documents=documents)
    if phase!='embed':
        rows = json.loads((src / 'queries.json').read_text())
        rows.sort(key=lambda q:int(q['id']))
        assert len(rows)==1200 and [int(q['id']) for q in rows]==list(range(1,1201))
        qids,qptr = flat_text([q['query'] for q in rows],256)
        qe = encode_flat(qids,qptr,8)
    # Search after embedding; keep only 1000 hit/chunk, vectors stay on Volume.
    done = token_count = offset = 0
    ran_parts = []
    embdir = dest / 'vectors'
    embdir.mkdir(exist_ok=True)
    for part in manifest['files']:
        guard()
        p = src / part['name']
        import hashlib
        vp = embdir / (p.stem+'.npz')
        expected_ids = np.arange(offset,offset+part['chunks'],dtype=np.int64)
        offset += part['chunks']
        if vp.exists() and phase in ('embed','search'):
            try:
                with np.load(vp) as z:
                    validate_vectors(z['vectors'],z['chunk_ids'],expected_ids)
                done += part['chunks']
                token_count += part['tokens']
                continue
            except (AssertionError,ValueError,KeyError,OSError):
                assert phase=='embed' and part['name'] in plan['run_parts'], 'Unaudited corrupt vector'
        assert phase!='search', 'Search requires every part valid; never embeds'
        assert hashlib.sha256(p.read_bytes()).hexdigest()==part['sha256']
        if plan is not None and part['name'] not in plan['run_parts']:
            audited = next(x for x in plan['good'] if x['name']==part['name'])
            assert hashlib.sha256(vp.read_bytes()).hexdigest()==audited['sha256']
            done += part['chunks']
            token_count += part['tokens']
            continue
        if vp.exists():
            assert plan is not None and part['name'] in {x['name'] for x in plan['bad']}, 'Refuse ambiguous resume; explicit audit required'
        z = np.load(p)
        ids,ptr,cids = z['ids'],z['ptr'],z['chunk_id']
        if selected is not None:
            ids,ptr,cids = select_flat(ids,ptr,cids,selected)
        if not len(cids):
            continue
        e = encode_flat(ids,ptr)
        validate_vectors(e,cids,expected_ids if scope=='full' else cids)
        token_count += len(ids)
        np.savez(vp,chunk_ids=cids,vectors=e)
        done += len(e)
        ran_parts.append(part['name'])
        print('EMBED',done,expected_chunks,scope,round(time.time()-t0,2),flush=True)
        vol.commit()
    assert done == expected_chunks and token_count==expected_tokens
    if phase!='search':
        (dest/'embedding_complete.json').write_text(json.dumps(dict(chunks=done,tokens=token_count,run_parts=ran_parts,elapsed_seconds=time.time()-t0),indent=2))
        vol.commit()
    report['embedding_and_startup_seconds'] = time.time()-t0
    if phase=='embed':
        report.update(chunks=done,tokens=token_count,ran_parts=ran_parts,total_seconds=time.time()-t0)
        report['usd_upper_function'] = report['total_seconds']*rate
        (dest/'embed_report.json').write_text(json.dumps(report,indent=2))
        vol.commit()
        return report
    del model
    torch.cuda.empty_cache()
    tq = torch.from_numpy(qe).cuda().float()
    top_s = torch.full((1200,1000),-float('inf'),device='cuda',dtype=torch.float32)
    top_c = torch.full((1200,1000),-1,device='cuda',dtype=torch.int64)
    s0 = time.time()
    # fp32 IP (TF32 disabled) preserves original dense cosine ranking to fp16 store precision.
    torch.backends.cuda.matmul.allow_tf32 = False
    for part in manifest['files']:
        guard()
        vp = embdir / (Path(part['name']).stem+'.npz')
        if not vp.exists():
            assert selected is not None
            continue
        z = np.load(vp)
        E = torch.from_numpy(z['vectors']).cuda().float()
        C = torch.from_numpy(z['chunk_ids']).cuda()
        for q in range(0,1200,100):
            sc = tq[q:q+100] @ E.T
            vals,ix = torch.topk(sc,k=min(1000,len(E)),dim=1,sorted=False)
            cs = C[ix]
            joined = torch.cat((top_s[q:q+100],vals),dim=1)
            joined_c = torch.cat((top_c[q:q+100],cs),dim=1)
            vals,ix = torch.topk(joined,k=1000,dim=1,sorted=True)
            top_s[q:q+100] = vals
            top_c[q:q+100] = torch.gather(joined_c,1,ix)
        del E,C
    assert bool(torch.isfinite(top_s).all()) and bool((top_c>=0).all())
    np.savez_compressed(dest / 'top1000.npz', qids=np.arange(1,1201,dtype=np.int32),
                        chunk_ids=top_c.cpu().numpy(),scores=top_s.cpu().numpy())
    import hashlib
    report['top1000_sha256'] = hashlib.sha256((dest/'top1000.npz').read_bytes()).hexdigest()
    report['top1000_bytes'] = (dest/'top1000.npz').stat().st_size
    report.update(chunks=done,tokens=token_count, search_seconds=time.time()-s0,total_seconds=time.time()-t0,
                  max_memory_gib=torch.cuda.max_memory_allocated()/2**30)
    report['usd_upper_function'] = report['total_seconds']*rate
    (dest / 'bulk_report.json').write_text(json.dumps(report,indent=2))
    vol.commit()
    return report


@app.local_entrypoint()
def main(stage: str='bench', seconds_budget: int=6000, gpu_kind: str='L40S', scope: str='full'):
    if stage=='bench':
        print(audit_volume.remote())
        # Sequential inference, minimum 2s scaledown; covered by 15s reserve.
        reports = [bench.remote(), bench_h100.remote()]
        best = min(reports,key=lambda r:r['predicted_bulk_usd'])
        summary = dict(best, comparisons=reports)
        print(summary)
    elif stage=='bulk':
        print((bulk_h100 if gpu_kind=='H100' else bulk).remote(seconds_budget,scope))
    elif stage=='audit-partial':
        print(audit_partial.remote())
    elif stage=='resume':
        raise RuntimeError('Unsafe local-entrypoint resume disabled: deploy app, then use dchunk_service.py embed')
    else:
        raise ValueError(stage)
