"""Create the resumable 2xT4 Exp144 QuerySpec→BM25→BGE notebook."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/23_exp144_queryspec_dense_2xt4_kaggle.ipynb"


def lines(value: str) -> list[str]:
    return [line + "\n" for line in value.strip("\n").splitlines()]


def markdown(value: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines(value)}


def code(value: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines(value)}


cells = [
    markdown(r"""
# Exp144 — QuerySpec expansion → BM25 candidate union → exact BGE-M3 rerank

This job changes candidate generation, then scores every candidate with the **original Vietnamese query**. QuerySpec canonical forms, atoms and translations can rescue vocabulary mismatch but cannot directly determine final rank.

Inputs:

1. Upload `data/kaggle_inputs/exp144_queryspec_dense_input.zip` as a private Kaggle dataset.
2. Attach the private active index dataset containing `chunks_metadata.parquet`, `emb_dense/`, and `rebuild_manifest.json`.
3. Enable two T4 GPUs. A one-GPU run is supported but slower.

Output is a small ranking archive. Build BTC submission variants locally with `scripts/build_exp144_docs_candidates.py`, preserving Exp143 chunks exactly.
"""),
    code(r"""
%pip install -q "FlagEmbedding==1.3.5" "transformers==4.51.3" "accelerate==1.6.0" "sentence-transformers==3.4.1" "huggingface-hub==0.30.2" "pyarrow>=18,<21" "pandas>=2.2,<3"
"""),
    code(r"""
from __future__ import annotations
import gc, hashlib, io, json, os, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

INPUT_ROOT=Path('/kaggle/input')
WORK=Path('/kaggle/working/exp144'); WORK.mkdir(parents=True,exist_ok=True)
MODEL_ID='BAAI/bge-m3'
RANK_K=400
CHECKPOINT_EVERY=25

packages=list(INPUT_ROOT.rglob('exp144_queryspec_dense_input.zip'))
unpacked=[p.parent for p in INPUT_ROOT.rglob('queryspec_candidates.parquet')
          if (p.parent/'query.parquet').exists()]
index_roots=[p.parent for p in INPUT_ROOT.rglob('chunks_metadata.parquet')
             if (p.parent/'emb_dense').is_dir() or list(p.parent.glob('emb_dense__ids_*.npy'))]
if len(packages)+len(unpacked)!=1: raise RuntimeError(f'Expected one Exp144 input, found {len(packages)} ZIP and {len(unpacked)} unpacked')
if len(index_roots)!=1: raise RuntimeError(f'Expected one active index, found {len(index_roots)}')
INDEX=index_roots[0]
if packages:
    INPUT=WORK/'input'; INPUT.mkdir(exist_ok=True)
    with zipfile.ZipFile(packages[0]) as z:
        if z.testzip() is not None: raise RuntimeError('Input package CRC failed')
        z.extractall(INPUT)
else:
    INPUT=unpacked[0]
queries=pd.read_parquet(INPUT/'query.parquet').sort_values('id').reset_index(drop=True)
specs=pd.read_parquet(INPUT/'query_specs.parquet').sort_values('query_id').reset_index(drop=True)
candidates=pd.read_parquet(INPUT/'queryspec_candidates.parquet',columns=['query_id','rank','doc_id'])
control_frame=pd.read_parquet(INPUT/'exp143_control_docs.parquet',columns=['query_id','rank','doc_id'])
if len(queries)!=1200 or len(specs)!=1200 or control_frame.query_id.nunique()!=1200:
    raise RuntimeError('Expected 1,200 query/spec/control rows')
if control_frame.duplicated(['query_id','doc_id']).any(): raise RuntimeError('Duplicate control docs')
control_docs={int(q):g.sort_values('rank').doc_id.astype(int).tolist() for q,g in control_frame.groupby('query_id')}
candidate_docs={int(q):g.sort_values('rank').doc_id.astype(int).tolist() for q,g in candidates.groupby('query_id')}
if set(candidate_docs)!=set(queries.id.astype(int)) or set(control_docs)!=set(queries.id.astype(int)):
    raise RuntimeError('Query coverage mismatch')
print({'queries':len(queries),'candidate_rows':len(candidates),'gpus':torch.cuda.device_count(),
       'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
"""),
    code(r"""
from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import snapshot_download
if not torch.cuda.is_available(): raise RuntimeError('GPU required')
model_path=snapshot_download(repo_id=MODEL_ID)
MODEL_REVISION=Path(model_path).name
encoder=BGEM3FlagModel(model_path,use_fp16=True)
t0=time.time()
encoded=encoder.encode(queries['query'].astype(str).tolist(),batch_size=8,max_length=1024,
                       return_dense=True,return_sparse=False,return_colbert_vecs=False)
qvec=np.asarray(encoded['dense_vecs'],dtype=np.float32)
qvec/=np.maximum(np.linalg.norm(qvec,axis=1,keepdims=True),1e-12)
del encoded,encoder; gc.collect(); torch.cuda.empty_cache()
print({'query_vectors':qvec.shape,'seconds':round(time.time()-t0,1),'model_revision':MODEL_REVISION})
"""),
    code(r"""
meta=pd.read_parquet(INDEX/'chunks_metadata.parquet',columns=['chunk_id','doc_id']).sort_values('chunk_id').reset_index(drop=True)
if not np.array_equal(meta.chunk_id.to_numpy(),np.arange(len(meta),dtype=np.int64)):
    raise RuntimeError('Chunk IDs are not contiguous')
doc_arr=meta.doc_id.to_numpy(dtype=np.int64)
doc_chunks=meta.groupby('doc_id',sort=False).chunk_id.apply(lambda x:x.to_numpy(dtype=np.int32)).to_dict()
if (INDEX/'emb_dense').is_dir():
    id_paths=sorted((INDEX/'emb_dense').glob('ids_*.npy')); vec_paths=sorted((INDEX/'emb_dense').glob('vec_*.npy'))
else:
    id_paths=sorted(INDEX.glob('emb_dense__ids_*.npy')); vec_paths=sorted(INDEX.glob('emb_dense__vec_*.npy'))
if not id_paths or len(id_paths)!=len(vec_paths): raise RuntimeError('Dense shard set incomplete')
device_count=max(torch.cuda.device_count(),1)
mapped=[]; offset=0
for ip,vp in zip(id_paths,vec_paths):
    ids=np.load(ip,mmap_mode='r'); vec=np.load(vp,mmap_mode='r')
    if vec.shape!=(len(ids),1024) or int(ids[0])!=offset or not np.array_equal(ids,np.arange(offset,offset+len(ids))):
        raise RuntimeError(f'Shard alignment failed: {ip.name}')
    mapped.append(vec); offset+=len(ids)
if offset!=len(meta): raise RuntimeError('Vector/metadata row mismatch')
# Consolidate the many on-disk shards into one contiguous range per GPU. This
# keeps exact vectors but reduces every query from ~23 tiny GPU gathers to 1–2.
all_vectors=np.concatenate(mapped,axis=0)
boundaries=np.linspace(0,len(all_vectors),device_count+1,dtype=np.int64)
vec_shards=[]
for part in range(device_count):
    lo,hi=map(int,boundaries[part:part+2]); device=f'cuda:{part}'
    tensor=torch.from_numpy(np.ascontiguousarray(all_vectors[lo:hi])).to(device,non_blocking=True)
    vec_shards.append((lo,hi,device,tensor))
del mapped,all_vectors; gc.collect()
print({'chunks':len(meta),'docs':len(doc_chunks),'gpu_ranges':[(x[0],x[1],x[2]) for x in vec_shards],
       'gpu_allocated_gb':[round(torch.cuda.memory_allocated(i)/1e9,2) for i in range(device_count)]})
"""),
    code(r"""
def score_chunks(chunk_ids,query_vector):
    output=np.empty(len(chunk_ids),dtype=np.float32)
    for lo,hi,device,vectors in vec_shards:
        mask=(chunk_ids>=lo)&(chunk_ids<hi)
        if not mask.any(): continue
        local=torch.as_tensor(chunk_ids[mask].astype(np.int64)-lo,device=device)
        selected=vectors.index_select(0,local).float()
        q=torch.as_tensor(query_vector,device=device,dtype=torch.float32)
        output[mask]=(selected@q).detach().cpu().numpy()
    if not np.isfinite(output).all(): raise RuntimeError('Non-finite scores')
    return output

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

fingerprint={
    'input_sha256':sha256(packages[0] if packages else INPUT/'queryspec_candidates.parquet'),
    'index_manifest_sha256':sha256(INDEX/'rebuild_manifest.json') if (INDEX/'rebuild_manifest.json').exists() else None,
    'model':MODEL_ID,'revision':MODEL_REVISION,'rank_k':RANK_K,
    'query_ids_sha256':hashlib.sha256(queries.id.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
}
checkpoint=WORK/'exp144_checkpoint.npz'
rankings={}; ranking_scores={}
if checkpoint.exists():
    with np.load(checkpoint,allow_pickle=False) as saved:
        if json.loads(str(saved['fingerprint'].item()))!=fingerprint: raise RuntimeError('Checkpoint fingerprint mismatch')
        done=saved['query_ids'].astype(np.int64); docs=saved['docs'].astype(np.int64); scores=saved['scores'].astype(np.float32)
    expected=queries.id.to_numpy(dtype=np.int64)[:len(done)]
    if not np.array_equal(done,expected) or docs.shape!=(len(done),RANK_K) or scores.shape!=docs.shape:
        raise RuntimeError('Checkpoint shape/order mismatch')
    for i,qid in enumerate(done): rankings[int(qid)]=docs[i].tolist(); ranking_scores[int(qid)]=scores[i].tolist()
    print({'resumed':len(done)})

def save_checkpoint():
    ids=np.asarray(list(rankings),dtype=np.int64)
    tmp=checkpoint.with_suffix('.tmp.npz')
    np.savez_compressed(tmp,query_ids=ids,docs=np.asarray(list(rankings.values()),dtype=np.int64),
                        scores=np.asarray(list(ranking_scores.values()),dtype=np.float32),
                        fingerprint=np.asarray(json.dumps(fingerprint,sort_keys=True)))
    os.replace(tmp,checkpoint)

t0=time.time(); start=len(rankings)
for qi in range(start,len(queries)):
    qid=int(queries.iloc[qi].id)
    current=control_docs[qid]
    pool=list(dict.fromkeys(current+candidate_docs[qid]))
    pool=[doc for doc in pool if doc in doc_chunks]
    if len(pool)<RANK_K: raise RuntimeError(f'Only {len(pool)} indexed candidates for {qid}')
    lengths=np.fromiter((len(doc_chunks[d]) for d in pool),dtype=np.int32,count=len(pool))
    chunk_ids=np.concatenate([doc_chunks[d] for d in pool]).astype(np.int32,copy=False)
    chunk_doc_index=np.repeat(np.arange(len(pool),dtype=np.int32),lengths)
    chunk_scores=score_chunks(chunk_ids,qvec[qi])
    doc_scores=np.full(len(pool),-np.inf,dtype=np.float32); np.maximum.at(doc_scores,chunk_doc_index,chunk_scores)
    order=np.argsort(-doc_scores,kind='stable')[:RANK_K]
    rankings[qid]=[pool[i] for i in order]
    ranking_scores[qid]=doc_scores[order].tolist()
    if (qi+1)%CHECKPOINT_EVERY==0 or qi+1==len(queries):
        save_checkpoint(); done=qi-start+1; elapsed=time.time()-t0
        print(f'{qi+1}/1200 | {done/max(elapsed,1e-6):.2f} q/s | ETA {(1200-qi-1)*elapsed/max(done,1)/60:.1f} min',flush=True)
"""),
    code(r"""
if len(rankings)!=1200: raise RuntimeError(f'Incomplete rankings: {len(rankings)}')
query_ids=np.asarray(list(rankings),dtype=np.int64)
docs=np.asarray(list(rankings.values()),dtype=np.int64)
scores=np.asarray(list(ranking_scores.values()),dtype=np.float32)
np.savez_compressed(WORK/'exp144_rankings.npz',query_ids=query_ids,docs=docs,scores=scores)
diagnostics=[]
for qid,row_docs,row_scores in zip(query_ids,docs,scores):
    current=set(control_docs[int(qid)])
    diagnostics.append({'query_id':int(qid),'current_docs':len(current),
                        'top60_overlap':sum(int(x) in current for x in row_docs[:60]),
                        'top90_overlap':sum(int(x) in current for x in row_docs[:90]),
                        'top130_overlap':sum(int(x) in current for x in row_docs[:130]),
                        'novel_in_top20':sum(int(x) not in current for x in row_docs[:20]),
                        'rank1_score':float(row_scores[0]),'rank400_score':float(row_scores[-1])})
pd.DataFrame(diagnostics).to_parquet(WORK/'exp144_diagnostics.parquet',index=False)
manifest={**fingerprint,'queries':1200,'ranking_k':RANK_K,
          'rankings_sha256':sha256(WORK/'exp144_rankings.npz'),
          'diagnostics_sha256':sha256(WORK/'exp144_diagnostics.parquet'),
          'purpose':'docs-axis candidate generation; preserve Exp143 chunks locally',
          'limitation':'No qrels; BTC scoring is required.'}
(WORK/'exp144_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
output=Path('/kaggle/working/exp144_queryspec_dense_outputs.zip')
with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for name in ['exp144_rankings.npz','exp144_diagnostics.parquet','exp144_manifest.json']:
        z.write(WORK/name,name)
with zipfile.ZipFile(output) as z:
    if z.testzip() is not None: raise RuntimeError('Output CRC failed')
print({'output':str(output),'bytes':output.stat().st_size,'sha256':sha256(output),
       'diagnostics':pd.DataFrame(diagnostics).describe().round(3).to_dict()})
"""),
    markdown(r"""
## After the Kaggle run

Download only `exp144_queryspec_dense_outputs.zip`. Do not download the model cache or private index. Build one docs-only submission at a time locally so every candidate keeps Exp143 chunks byte-for-byte and BTC can attribute any change to the document axis.
"""),
]

notebook = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "kaggle": {"accelerator": "gpu", "dataSources": [], "dockerImageVersionId": None,
                   "isGpuEnabled": True, "isInternetEnabled": True, "language": "python",
                   "sourceType": "notebook"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
if OUTPUT.exists():
    raise FileExistsError(OUTPUT)
OUTPUT.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
print(OUTPUT)
