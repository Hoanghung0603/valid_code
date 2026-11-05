"""Generate a private two-T4 BGE reranker notebook for exact P2a chunk windows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nbformat as nbf

from create_p1_full_reranker_kaggle_notebook import load_pilot_worker


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "BAAI/bge-reranker-v2-m3"


def code(source: str):
    compile(source, "<p2a-notebook-cell>", "exec")
    return nbf.v4.new_code_cell(source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scope", choices=("pilot", "full"), required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or ROOT / f"notebooks/20_p2a_bge_chunk_{args.scope}_2xt4_kaggle.ipynb"
    worker = load_pilot_worker()
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU", "kaggle": {"accelerator": "gpu", "dataSources": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    notebook.cells = [nbf.v4.new_markdown_cell(
        f"""# P2a — BGE reranking of existing chunks ({args.scope}, 2×T4)

Scores deterministic windows cut from the *exact* P1-submitted chunk text.
Outputs scores only, never replacement text or a BTC submission. Required
private dataset: `ptquanh/r2ai-p2a-chunk-{args.scope}-v1`.
Optional Kaggle Secret: `HF_TOKEN`. Model: `{MODEL_ID}` (<15B parameters)."""
    ), nbf.v4.new_code_cell(
        "%pip install -q --no-cache-dir 'transformers>=4.51,<5' 'accelerate>=1.6,<2' 'huggingface-hub>=0.30,<1' 'pyarrow>=18,<22' 'pandas>=2.2,<3'"
    )]
    notebook.cells.append(code(f"""import hashlib, json, os, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

SCOPE={args.scope!r}
MODEL_KEY='bge'
MODEL_ID={MODEL_ID!r}
BATCH_SIZE=16
MAX_LENGTH=1024
INSTRUCTION='Judge whether this exact retrieved chunk contains evidence that answers the multilingual medical question.'
WORK=Path('/kaggle/working/p2a_chunk_{args.scope}')
WORK.mkdir(parents=True,exist_ok=True)
try:
    from kaggle_secrets import UserSecretsClient
    token=UserSecretsClient().get_secret('HF_TOKEN')
    if token: os.environ['HF_TOKEN']=token
except Exception: pass

def sha256_file(path,block=4*1024*1024):
    digest=hashlib.sha256()
    with open(path,'rb') as source:
        while piece:=source.read(block): digest.update(piece)
    return digest.hexdigest()

assert torch.cuda.device_count()==2,torch.cuda.device_count()
input_root=Path('/kaggle/input')
pair_paths=list(input_root.rglob('p2a_pairs.parquet'))
chunk_paths=list(input_root.rglob('p2a_chunks.parquet'))
manifest_paths=list(input_root.rglob('p2a_manifest.json'))
if not(len(pair_paths)==len(chunk_paths)==len(manifest_paths)==1):
    raise RuntimeError({{'pairs':pair_paths,'chunks':chunk_paths,'manifests':manifest_paths}})
manifest=json.loads(manifest_paths[0].read_text(encoding='utf-8'))
assert manifest['schema_version']=='p2a_chunk_windows_v1' and manifest['scope']==SCOPE
for path in (pair_paths[0],chunk_paths[0]):
    assert sha256_file(path)==manifest['artifacts'][path.name]['sha256'],path
pairs=pd.read_parquet(pair_paths[0]); chunks=pd.read_parquet(chunk_paths[0])
assert len(pairs)==manifest['validation']['pairs'] and len(chunks)==manifest['validation']['chunks']
assert pairs.pair_id.is_unique and chunks[['query_id','chunk_ordinal']].duplicated().sum()==0
assert pairs.query_id.nunique()==manifest['validation']['queries']
input_path=WORK/'p2a_pairs.parquet'
pairs.to_parquet(input_path,index=False,compression='zstd')
print({{'gpus':[torch.cuda.get_device_name(i) for i in range(2)],'scope':SCOPE,
       'queries':pairs.query_id.nunique(),'chunks':len(chunks),'windows':len(pairs),
       'control_hash':manifest['control_zip_sha256']}},flush=True)"""))
    notebook.cells.append(code(
        "WORKER = " + repr(worker) + "\nworker_path=WORK/'p2a_worker.py'\nworker_path.write_text(WORKER,encoding='utf-8')\nprint({'worker_chars':len(WORKER)})"
    ))
    notebook.cells.append(code("""processes=[]; logs=[]
for gpu in range(2):
    out=WORK/'checkpoints'/f'gpu{gpu}'; out.mkdir(parents=True,exist_ok=True)
    log_path=WORK/f'bge_gpu{gpu}.log'; log=log_path.open('w',encoding='utf-8'); logs.append((log,log_path))
    cmd=[sys.executable,str(worker_path),'--input',str(input_path),'--out',str(out),
         '--model-key',MODEL_KEY,'--model-id',MODEL_ID,'--gpu',str(gpu),
         '--batch-size',str(BATCH_SIZE),'--max-length',str(MAX_LENGTH),'--instruction',INSTRUCTION]
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu); env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env))
while any(process.poll() is None for process in processes):
    time.sleep(60)
    status={'returncodes':[process.poll() for process in processes]}
    for _,path in logs:
        if path.exists(): status[path.name]=path.read_text(encoding='utf-8',errors='replace').splitlines()[-1:]
    print(status,flush=True)
for log,_ in logs: log.close()
for process,(_,path) in zip(processes,logs):
    if process.returncode!=0:
        raise RuntimeError(f'P2a BGE worker failed ({process.returncode}):\\n'+path.read_text(encoding='utf-8',errors='replace')[-8000:])
scores=pd.concat([pd.read_parquet(WORK/'checkpoints'/f'gpu{gpu}'/f'bge_gpu{gpu}.parquet') for gpu in range(2)],ignore_index=True)
assert len(scores)==len(pairs) and scores.pair_id.is_unique
assert set(scores.pair_id)==set(pairs.pair_id)
assert scores.model_revision.nunique()==1 and np.isfinite(scores.raw_score).all()
print({'scored_windows':len(scores),'revision':scores.model_revision.iloc[0]})"""))
    notebook.cells.append(code("""scored=pairs.drop(columns=['query_text','passage_text']).merge(
    scores[['pair_id','raw_score','normalized_score','input_tokens','model_revision','gpu_shard']],
    on='pair_id',validate='one_to_one')
pair_output=WORK/'p2a_pair_scores.parquet'
scored.to_parquet(pair_output,index=False,compression='zstd')
aggregated=scored.groupby(['query_id','chunk_ordinal','doc_id','chunk_sha256'],as_index=False).agg(
    bge_chunk_score=('normalized_score','max'),bge_chunk_raw=('raw_score','max'),
    windows_scored=('pair_id','size'),max_input_tokens=('input_tokens','max'))
chunk_scores=chunks.merge(aggregated,on=['query_id','chunk_ordinal','doc_id','chunk_sha256'],validate='one_to_one')
assert len(chunk_scores)==len(chunks) and (chunk_scores.window_count==chunk_scores.windows_scored).all()
chunk_scores['bge_chunk_rank']=chunk_scores.groupby('query_id')['bge_chunk_score'].rank(method='first',ascending=False).astype(int)
chunk_output=WORK/'p2a_chunk_scores.parquet'
chunk_scores.to_parquet(chunk_output,index=False,compression='zstd')
validation={'schema_version':'p2a_chunk_scores_v1','scope':SCOPE,'model_key':MODEL_KEY,'model_id':MODEL_ID,
 'model_revision':str(scores.model_revision.iloc[0]),'queries':int(chunk_scores.query_id.nunique()),
 'chunks':len(chunk_scores),'pairs':len(scored),'nonfinite':int((~np.isfinite(chunk_scores.bge_chunk_score)).sum()),
 'control_zip_sha256':manifest['control_zip_sha256'],
 'input_pairs_sha256':manifest['artifacts']['p2a_pairs.parquet']['sha256'],
 'input_chunks_sha256':manifest['artifacts']['p2a_chunks.parquet']['sha256']}
(WORK/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
run_manifest={'schema_version':'p2a_chunk_reranker_run_v1','input_manifest':manifest,'validation':validation,
 'aggregation':'max normalized window score per exact submitted chunk','max_length':MAX_LENGTH,
 'outputs':{path.name:sha256_file(path) for path in (pair_output,chunk_output)}}
(WORK/'run_manifest.json').write_text(json.dumps(run_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
output_zip=Path('/kaggle/working')/f'p2a_bge_chunk_{SCOPE}_scores.zip'
members=['p2a_pair_scores.parquet','p2a_chunk_scores.parquet','validation.json','run_manifest.json']
with zipfile.ZipFile(output_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for name in members: archive.write(WORK/name,arcname=name)
assert zipfile.ZipFile(output_zip).testzip() is None
print(validation); print({'output':str(output_zip),'bytes':output_zip.stat().st_size,'sha256':sha256_file(output_zip)})"""))
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    json.loads(output.read_text(encoding="utf-8"))
    print({"notebook": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
           "cells": len(notebook.cells)})


if __name__ == "__main__":
    main()
