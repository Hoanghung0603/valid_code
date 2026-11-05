"""Generate a one-model, query-partitioned P1 full-reranker Kaggle notebook."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
PILOT_NOTEBOOK = ROOT / "notebooks/17_p1_reranker_pilot_2xt4_kaggle.ipynb"
MODELS = {
    "bge": {"id": "BAAI/bge-reranker-v2-m3", "batch": 16},
    "qwen4b": {"id": "Qwen/Qwen3-Reranker-4B", "batch": 4},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--part-index", type=int, required=True, help="Zero-based query partition")
    parser.add_argument("--part-count", type=int, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def code(value: str):
    compile(value, "<notebook-cell>", "exec")
    return nbf.v4.new_code_cell(value)


def load_pilot_worker() -> str:
    notebook = nbf.read(PILOT_NOTEBOOK, as_version=4)
    for cell in notebook.cells:
        source = cell.source
        if cell.cell_type == "code" and source.startswith("WORKER = "):
            expression = source.splitlines()[0].removeprefix("WORKER = ")
            worker = ast.literal_eval(expression)
            compile(worker, "<p1-worker>", "exec")
            return worker
    raise RuntimeError("Could not recover the audited P1 worker from the pilot notebook")


def main() -> None:
    args = parse_args()
    if args.part_count < 1 or not 0 <= args.part_index < args.part_count:
        raise ValueError("Require 0 <= part-index < part-count")
    if args.model == "qwen4b" and args.part_count < 4:
        raise ValueError("Qwen4B full run needs at least four partitions to stay below the 12-hour Kaggle session limit")
    spec = MODELS[args.model]
    output = args.output or ROOT / f"notebooks/19_p1_full_{args.model}_part{args.part_index + 1:02d}of{args.part_count:02d}_2xt4_kaggle.ipynb"
    worker = load_pilot_worker()
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "kaggle": {"accelerator": "gpu", "dataSources": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    cells = [nbf.v4.new_markdown_cell(
        f"""# P1 full reranker — {args.model}, part {args.part_index + 1}/{args.part_count}, 2×T4

Runs exactly one pilot-selected reranker on a deterministic query-disjoint
partition of the private P1 full pack. It emits scores only; no competition
submission is created. Required dataset: `ptquanh/r2ai-p1-reranker-full-v1`."""
    )]
    cells.append(nbf.v4.new_code_cell(
        """%pip install -q --no-cache-dir 'transformers>=4.51,<5' 'accelerate>=1.6,<2' 'huggingface-hub>=0.30,<1' 'pyarrow>=18,<22' 'pandas>=2.2,<3'"""
    ))
    cells.append(code(
        f"""import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

MODEL_KEY={args.model!r}
MODEL_ID={spec['id']!r}
BATCH_SIZE={spec['batch']}
PART_INDEX={args.part_index}
PART_COUNT={args.part_count}
INSTRUCTION='Given a multilingual medical question, retrieve passages that contain information needed to answer the question while respecting clinical intent, population, treatment, and negation constraints.'
WORK=Path('/kaggle/working/p1_full')
WORK.mkdir(parents=True,exist_ok=True)
INPUT_ROOT=Path('/kaggle/input')

try:
    from kaggle_secrets import UserSecretsClient
    token=UserSecretsClient().get_secret('HF_TOKEN')
    if token: os.environ['HF_TOKEN']=token
except Exception: pass

def sha256_file(path,block=4*1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while b:=f.read(block): h.update(b)
    return h.hexdigest()

assert torch.cuda.device_count()==2, torch.cuda.device_count()
pair_candidates=list(INPUT_ROOT.rglob('p1_pairs.parquet'))
manifest_candidates=list(INPUT_ROOT.rglob('p1_manifest.json'))
query_candidates=list(INPUT_ROOT.rglob('p1_queries.parquet'))
if not (len(pair_candidates)==len(manifest_candidates)==len(query_candidates)==1):
    raise RuntimeError({{'pairs':pair_candidates,'manifests':manifest_candidates,'queries':query_candidates}})
manifest=json.loads(manifest_candidates[0].read_text(encoding='utf-8'))
assert manifest['scope']=='full' and manifest['validation']['queries']==1200
pairs=pd.read_parquet(pair_candidates[0]); queries=pd.read_parquet(query_candidates[0])
all_qids=sorted(queries.query_id.astype(int).unique())
owned={{qid for number,qid in enumerate(all_qids) if number%PART_COUNT==PART_INDEX}}
pairs=pairs[pairs.query_id.isin(owned)].copy().reset_index(drop=True)
queries=queries[queries.query_id.isin(owned)].copy().reset_index(drop=True)
assert pairs.groupby(['query_id','doc_id']).ngroups==len(owned)*216
input_path=WORK/'p1_pairs_partition.parquet'
pairs.to_parquet(input_path,index=False,compression='zstd')
print({{'gpus':[torch.cuda.get_device_name(i) for i in range(2)],'model':MODEL_KEY,'part':f'{{PART_INDEX+1}}/{{PART_COUNT}}','queries':len(owned),'pairs':len(pairs),'full_pairs_sha256':manifest['artifacts']['p1_pairs.parquet']['sha256']}})"""
    ))
    cells.append(code(
        "WORKER = " + repr(worker) + "\nworker_path=WORK/'p1_worker.py'\nworker_path.write_text(WORKER,encoding='utf-8')\nprint({'worker_chars':len(WORKER)})"
    ))
    cells.append(code(
        """processes=[]; logs=[]
for gpu in range(2):
    out=WORK/'checkpoints'/f'gpu{gpu}'; out.mkdir(parents=True,exist_ok=True)
    log_path=WORK/f'{MODEL_KEY}_gpu{gpu}.log'; log=log_path.open('w',encoding='utf-8'); logs.append((log,log_path))
    cmd=[sys.executable,str(worker_path),'--input',str(input_path),'--out',str(out),'--model-key',MODEL_KEY,'--model-id',MODEL_ID,'--gpu',str(gpu),'--batch-size',str(BATCH_SIZE),'--max-length','1024','--instruction',INSTRUCTION]
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu); env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
    processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env))
while any(process.poll() is None for process in processes):
    time.sleep(60)
    status={'model':MODEL_KEY,'returncodes':[process.poll() for process in processes]}
    for _,path in logs:
        if path.exists(): status[path.name]=path.read_text(encoding='utf-8',errors='replace').splitlines()[-1:]
    print(status,flush=True)
for log,_ in logs: log.close()
for process,(_,path) in zip(processes,logs):
    if process.returncode!=0: raise RuntimeError(f'{MODEL_KEY} worker failed ({process.returncode}):\\n'+path.read_text(encoding='utf-8',errors='replace')[-8000:])
frames=[pd.read_parquet(WORK/'checkpoints'/f'gpu{gpu}'/f'{MODEL_KEY}_gpu{gpu}.parquet') for gpu in range(2)]
scores=pd.concat(frames,ignore_index=True).sort_values(['query_id','doc_id','passage_rank']).reset_index(drop=True)
assert len(scores)==len(pairs) and scores.pair_id.nunique()==len(pairs) and np.isfinite(scores.raw_score).all()"""
    ))
    cells.append(code(
        """base=pairs[['pair_id','query_id','doc_id','doc_rank','owner_doc','passage_rank','language','dense_score']]
scored=base.merge(scores[['pair_id','raw_score','normalized_score','input_tokens','model_revision']],on='pair_id',validate='one_to_one')
score_col=f'{MODEL_KEY}_score'; raw_col=f'{MODEL_KEY}_raw'; rank_col=f'{MODEL_KEY}_rank'
scored=scored.rename(columns={'normalized_score':score_col,'raw_score':raw_col})
pair_output=WORK/'p1_pair_scores.parquet'; scored.to_parquet(pair_output,index=False,compression='zstd')
docs=scored.groupby(['query_id','doc_id','doc_rank','owner_doc'],as_index=False).agg(
    **{score_col:(score_col,'max'),raw_col:(raw_col,'max')},passage_count=('pair_id','size'),language=('language','first'))
docs[rank_col]=docs.groupby('query_id')[score_col].rank(method='first',ascending=False).astype(int)
doc_output=WORK/'p1_doc_scores.parquet'; docs.to_parquet(doc_output,index=False,compression='zstd')
validation={
 'model_key':MODEL_KEY,'model_id':MODEL_ID,'model_revision':str(scores.model_revision.iloc[0]),
 'part_index':PART_INDEX,'part_count':PART_COUNT,'queries':docs.query_id.nunique(),'query_docs':len(docs),'pairs':len(scored),
 'query_docs_min':int(docs.groupby('query_id').size().min()),'query_docs_max':int(docs.groupby('query_id').size().max()),
 'nonfinite':int((~np.isfinite(scored[raw_col])).sum()),'full_input_pairs_sha256':manifest['artifacts']['p1_pairs.parquet']['sha256'],
}
(WORK/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
run_manifest={'schema_version':'p1_full_reranker_part_v1','input_manifest':manifest,'validation':validation,'instruction':INSTRUCTION}
(WORK/'run_manifest.json').write_text(json.dumps(run_manifest,ensure_ascii=False,indent=2),encoding='utf-8')
zip_name=f'p1_full_{MODEL_KEY}_part{PART_INDEX+1:02d}of{PART_COUNT:02d}.zip'; output_zip=Path('/kaggle/working')/zip_name
members=['p1_pair_scores.parquet','p1_doc_scores.parquet','validation.json','run_manifest.json']
with zipfile.ZipFile(output_zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for name in members: archive.write(WORK/name,arcname=name)
assert zipfile.ZipFile(output_zip).testzip() is None
print(validation); print({'output':str(output_zip),'bytes':output_zip.stat().st_size,'sha256':sha256_file(output_zip)})"""
    ))
    notebook["cells"] = cells
    output.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, output)
    json.loads(output.read_text(encoding="utf-8"))
    print({"notebook": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "cells": len(cells)})


if __name__ == "__main__":
    main()
