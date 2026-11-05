"""vi41 retrieval only, reuses immutable dense-chunk vectors; never reembeds corpus."""
import modal
image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.6.0', 'transformers==4.51.3', 'numpy==1.26.4',
                 'huggingface_hub==0.30.2', 'safetensors==0.5.3', 'sentencepiece==0.2.0')
    .env({'HF_HOME':'/hf', 'HF_HUB_CACHE':'/hf', 'TOKENIZERS_PARALLELISM':'false'})
    .run_commands("python -c \"from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', allow_patterns=['config.json','pytorch_model.bin','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model'])\""))

vol = modal.Volume.from_name('r2ai-runner', create_if_missing=False)
app = modal.App('r2ai-vi41-search', image=image)


def guarded(job, function):
    import json
    from pathlib import Path
    flag = Path('/vol/out/vi41/search_init_failed.json')
    assert not flag.exists(), 'VI41 INIT GUARD: '+str(flag)
    try:
        return function()
    except Exception as exc:
        flag.parent.mkdir(parents=True, exist_ok=True)
        flag.write_text(json.dumps({'job':job, 'error':repr(exc)}))
        vol.commit()
        raise


@app.function(cpu=1, memory=2048, timeout=120, retries=0, volumes={'/vol':vol})
def ping():
    def run():
        import json
        from pathlib import Path
        import numpy
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
        report = {'numpy':str(numpy.__version__), 'torch':str(torch.__version__),
                  'transformers':str(transformers.__version__), 'gpu':False}
        Path('/vol/out/vi41/search_ping.json').write_text(json.dumps(report))
        vol.commit()
        return report
    return guarded('ping', run)


@app.function(gpu='H100', cpu=2, memory=16384, timeout=600,
              scaledown_window=2, max_containers=1, retries=0, volumes={'/vol':vol})
def search():
    from pathlib import Path
    assert Path('/vol/out/vi41/search_ping.json').exists(), 'CPU ping required'
    return guarded('search', run_search)


def run_search():
    import hashlib
    import json
    import time
    from pathlib import Path
    import numpy as np
    import torch
    from transformers import AutoModel, AutoTokenizer
    start = time.monotonic()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    dest = Path('/vol/out/vi41')
    rows = [json.loads(l) for l in (dest/'rewrites.jsonl').read_text().splitlines() if l.strip()]
    assert len(rows)==1200
    variants = [(int(r['id']),t) for r in rows for t in r['subqueries']+r['hyde']]
    tok = AutoTokenizer.from_pretrained('BAAI/bge-m3')
    model = AutoModel.from_pretrained('BAAI/bge-m3',torch_dtype=torch.float16,
                                     attn_implementation='sdpa').cuda().eval()
    qe = np.zeros((len(variants),1024),dtype=np.float16)
    def guard():
        assert time.monotonic()-start<540,'VI41 SEARCH WATCHDOG'
    with torch.inference_mode():
        for i in range(0,len(variants),32):
            guard()
            x = tok([t for _,t in variants[i:i+32]],padding=True,truncation=True,max_length=512,return_tensors='pt').to('cuda')
            h = model(**x).last_hidden_state[:,0]
            qe[i:i+32] = torch.nn.functional.normalize(h.float(),dim=-1).half().cpu().numpy()
    del model
    torch.cuda.empty_cache()
    manifest = json.loads(Path('/vol/in/dchunk/manifest.json').read_text())
    Q = torch.from_numpy(qe).cuda().float()
    top_s = torch.full((len(qe),1000),-float('inf'),device='cuda')
    top_c = torch.full((len(qe),1000),-1,device='cuda',dtype=torch.int64)
    for part in manifest['files']:
        guard()
        p = Path('/vol/out/dchunk/vectors')/(Path(part['name']).stem+'.npz')
        with np.load(p) as z:
            E = torch.from_numpy(z['vectors']).cuda().float()
            C = torch.from_numpy(z['chunk_ids']).cuda()
        assert len(E)==part['chunks'] and bool(torch.isfinite(E).all())
        for j in range(0,len(qe),100):
            guard()
            scores = Q[j:j+100] @ E.T
            vals,ix = scores.topk(min(1000,len(E)),dim=1)
            joined = torch.cat([top_s[j:j+100],vals],dim=1)
            ids = torch.cat([top_c[j:j+100],C[ix]],dim=1)
            vals,ix = joined.topk(1000,dim=1)
            top_s[j:j+100],top_c[j:j+100]=vals,ids.gather(1,ix)
        del E,C
    assert bool(torch.isfinite(top_s).all()) and bool((top_c>=0).all())
    target=dest/'dchunk_variants.npz'
    np.savez_compressed(target,qids=np.asarray([q for q,t in variants],dtype=np.int32),
        chunk_ids=top_c.cpu().numpy(),scores=top_s.cpu().numpy())
    report=dict(variants=len(variants),seconds=time.monotonic()-start,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        bytes=target.stat().st_size,query_max_length=512,parts=len(manifest['files']),corpus_reembedded=False)
    (dest/'dchunk_search_report.json').write_text(json.dumps(report,indent=2))
    vol.commit()
    return report
