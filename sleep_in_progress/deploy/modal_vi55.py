"""vi55 dedicated volumes; CPU-only downloads, offline fp16 GPU, no retries."""
from pathlib import Path
import os
import modal

MODEL='Qwen/Qwen3-Embedding-8B'
REVISION='1d8ad4ca9b3dd8059ad90a75d4983776a23d44af'
image=(modal.Image.debian_slim(python_version='3.11')
       .pip_install('torch==2.9.1','transformers==4.57.6','zstandard==0.23.0')
       .env({'VI55_RECOVERY':os.environ.get('VI55_RECOVERY','0')})
       .add_local_file(str(Path(__file__).resolve().parents[1]/'scripts/vi55.py'),'/root/vi55.py'))
data=modal.Volume.from_name('r2ai-vi55-data',create_if_missing=True)
weights=modal.Volume.from_name('r2ai-vi55-weights',create_if_missing=True)
old=None if os.environ.get('VI55_RECOVERY')=='1' else modal.Volume.from_name('r2ai-vi51-data',create_if_missing=False)
search_volumes={'/data':data,**({'/old':old} if old else {})}
app=modal.App('r2ai-vi55',image=image)


@app.function(gpu='H100',cpu=4,memory=65536,timeout=240,retries=0,max_containers=1,scaledown_window=2,
              volumes=search_volumes)
def retrieve(job):
    import json,sys,time,numpy as np,torch
    sys.path.insert(0,'/root')
    from vi55 import read,write,fingerprint
    start=time.monotonic();data.reload()
    if Path('/old').exists():old.reload()
    torch.backends.cuda.matmul.allow_tf32=False
    torch.set_float32_matmul_precision('highest')
    spec=read('/data/retrieval_spec.json');allowed=set(spec['allowed_ids'])
    assert fingerprint('/data/query_embeddings.npz')==spec['query']
    qz=np.load('/data/query_embeddings.npz');qids=list(map(int,qz['ids']))
    assert len(qids)==1200 and set(qids)==set(range(1,1201))
    assert qz['emb'].shape==(1200,4096) and np.isfinite(qz['emb']).all()
    Q=torch.from_numpy(qz['emb'].astype(np.float32)).cuda()
    Q=torch.nn.functional.normalize(Q,p=2,dim=1)
    targets=read('/data/vi53_thresholds.json')
    targetids=[];thresholds=[]
    for qi,q in enumerate(qids):
        ids=[int(d) for d in spec['doclists'][str(q)] if int(d) in allowed]
        # CPU float64 dot of identical fp16 vectors; 1e-6 competition rank resolution.
        sc=torch.tensor([targets[str(q)][str(d)] for d in ids],device='cuda')
        targetids.append(torch.tensor(ids,device='cuda'));thresholds.append(sc)
    max_targets=max(map(len,thresholds))
    threshold_matrix=torch.full((len(qids),max_targets),float('inf'),device='cuda')
    counts=torch.zeros((len(qids),max_targets),dtype=torch.int64,device='cuda')
    for qi,th in enumerate(thresholds):threshold_matrix[qi,:len(th)]=th
    bests=torch.full((len(qids),1000),-float('inf'),device='cuda');besti=torch.full((len(qids),1000),-1,dtype=torch.int64,device='cuda')
    seen=set();files=[Path(p) for p in spec['files']]
    meta=[]
    for path in files:
        assert fingerprint(path)==spec['files'][str(path)]
        z=np.load(path);mask=np.asarray([int(d) in allowed for d in z['ids']]);ids=z['ids'][mask];emb=z['emb'][mask]
        assert not seen.intersection(map(int,ids));seen.update(map(int,ids))
        assert emb.dtype==np.float16 and np.isfinite(emb).all()
        assert np.max(abs(np.linalg.norm(emb.astype(np.float32),axis=1)-1))<.001
        D=torch.from_numpy(emb.astype(np.float32)).cuda();D=torch.nn.functional.normalize(D,p=2,dim=1)
        di=torch.from_numpy(ids.astype(np.int64)).cuda()
        for q0 in range(0,len(qids),32):
            sc=Q[q0:q0+32]@D.T
            vals,ix=torch.topk(sc,min(1000,len(ids)),dim=1,sorted=False)
            ss=torch.cat((bests[q0:q0+len(sc)],vals),dim=1)
            ii=torch.cat((besti[q0:q0+len(sc)],di[ix]),dim=1)
            sv,si=torch.topk(ss,1000,dim=1,sorted=True)
            bests[q0:q0+len(sc)]=sv;besti[q0:q0+len(sc)]=ii.gather(1,si)
            # Exact competition ranks at 1e-6 cosine resolution, ties share rank.
            asc=torch.sort(torch.round(sc*1e6)/1e6,dim=1).values.contiguous()
            counts[q0:q0+len(sc)]+=len(ids)-torch.searchsorted(asc,threshold_matrix[q0:q0+len(sc)].contiguous(),right=True)
        meta.append(dict(path=str(path),documents=len(ids)))
        print('SEARCHED '+str(path)+' '+str(len(seen)),flush=True)
    dest=Path('/data/search')/job;dest.mkdir(parents=True,exist_ok=True)
    scores=bests.cpu().numpy();ids=besti.cpu().numpy()
    top={str(q):[[int(d),float(s)] for d,s in zip(ids[i],scores[i]) if d>=0] for i,q in enumerate(qids)}
    ranks={str(q):{str(int(d)):int(c) for d,c in zip(targetids[i].cpu().numpy(),counts[i,:len(targetids[i])].cpu().numpy())} for i,q in enumerate(qids)}
    import zstandard as zstd
    with zstd.open(dest/'top1000.json.zst','wt',encoding='utf-8') as f:json.dump(top,f)
    write(dest/'rank_counts.json',ranks)
    write(dest/'seen_ids.json',sorted(seen))
    report=dict(documents=len(seen),files=meta,query=spec['query'],function_seconds=time.monotonic()-start,rank_convention='competition; cosine rounded1e-6; count strictly greater',
                outputs={p.name:fingerprint(p) for p in dest.iterdir()})
    write(dest/'report.json',report);data.commit();return report


@app.function(cpu=2,memory=4096,timeout=3600,retries=0,volumes={'/weights':weights})
def download_weights():
    import json,time
    from huggingface_hub import snapshot_download
    start=time.monotonic(); snapshot_download(MODEL,revision=REVISION,local_dir='/weights/model')
    result=dict(model=MODEL,revision=REVISION,seconds=time.monotonic()-start)
    Path('/weights/complete.json').write_text(json.dumps(result));weights.commit();return result


@app.function(cpu=2,memory=8192,timeout=3600,retries=0,volumes=search_volumes)
def audit_reuse():
    import json,sys,numpy as np
    sys.path.insert(0,'/root')
    from vi55 import fingerprint
    files={}; seen=set()
    for p in sorted(Path('/old/out/b').glob('fp16_bulk_b_*/*.emb.npz')):
        z=np.load(p); ids=list(map(int,z['ids'])); e=z['emb']
        assert e.dtype==np.float16 and e.shape==(len(ids),4096) and np.isfinite(e).all()
        assert np.max(abs(np.linalg.norm(e.astype(np.float32),axis=1)-1))<.001
        assert not seen.intersection(ids);seen.update(ids)
        files[str(p.relative_to('/old'))]=dict(**fingerprint(p),documents=len(ids))
    result=dict(files=files,documents=len(seen),ids=sorted(seen),read_only=True)
    Path('/data/reuse_audit.json').write_text(json.dumps(result));data.commit();return result


@app.function(cpu=4,memory=16384,timeout=1800,retries=0,volumes={**search_volumes,'/weights':weights})
def cpu_ping_inventory():
    import sys,numpy as np,torch,transformers
    sys.path.insert(0,'/root')
    from vi55 import read,write,fingerprint,audit_embedding,iterz
    data.reload();weights.reload()
    if Path('/old').exists():old.reload()
    prep=read('/data/prepare_report.json')
    input_files={}
    for p in sorted(Path('/data/in').glob('docs55.part*.jsonl.zst')):
        expected={k:prep['parts'][p.name][k] for k in ('bytes','sha256')}
        assert fingerprint(p)==expected
        input_files[p.name]=expected
    files={};seen=set();parts=[]
    for p in sorted(Path('/data/out').glob('bulk*/*.emb.npz')):
        part=p.name[:-len('.emb.npz')]
        source=Path('/data/in')/part
        assert source.exists()
        audit=audit_embedding(p,[r['id'] for r in iterz(source)])
        assert not seen.intersection(audit['ids']);seen.update(audit['ids'])
        files[str(p)]={k:audit[k] for k in ('bytes','sha256')};parts.append(part)
    if Path('/old').exists():
        previous=read('/data/reuse_audit.json')
        for name,meta in previous['files'].items():
            p=Path('/old')/name
            assert fingerprint(p)=={k:meta[k] for k in ('bytes','sha256')}
            with np.load(p) as z:ids=z['ids'].copy()
            audit=audit_embedding(p,ids)
            assert not seen.intersection(audit['ids']);seen.update(audit['ids'])
            files[str(p)]={k:audit[k] for k in ('bytes','sha256')}
    result=dict(torch=str(torch.__version__),transformers=transformers.__version__,inputs=input_files,
                weights=read('/weights/complete.json'),files=files,parts=parts,ids=sorted(seen))
    write('/data/inventory.json',result);data.commit();return result


@app.function(gpu='H100',cpu=4,memory=65536,timeout=3700,retries=0,max_containers=4,scaledown_window=2,
              volumes={'/data':data,'/weights':weights},
              env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
def work(mode,job,parts,watchdog,doc_len):
    import json,sys,time,uuid
    sys.path.insert(0,'/root')
    from vi55 import gpu,write
    data.reload(); weights.reload()
    dest=Path('/data/out')/job;dest.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    try:
        return gpu(mode,Path('/data/in'),dest,parts,watchdog,doc_len,data.commit)
    finally:
        wall=dict(seconds=time.monotonic()-start)
        write(dest/('function_wall.'+uuid.uuid4().hex+'.json'),wall)
        write(dest/'function_wall.json',wall);data.commit()
