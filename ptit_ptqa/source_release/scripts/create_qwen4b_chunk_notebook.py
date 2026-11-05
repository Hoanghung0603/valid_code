"""Generate Kaggle 2xT4 notebooks for original-chunk Qwen4B scoring."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts/qwen4b_chunk_score_worker.py"
MODEL_ID = "Qwen/Qwen3-Reranker-4B"
REVISION = "22e683669bc0f0bd69640a1354a6d0aebcfeede5"
INSTRUCTION = "Given a multilingual medical question, retrieve passages that contain information needed to answer the question while respecting clinical intent, population, treatment, and negation constraints."


def code(value: str):
    compile(value, "<cell>", "exec"); return nbf.v4.new_code_cell(value)


def generate(scope: str, part_index: int, part_count: int, output: Path) -> None:
    worker = WORKER.read_text(encoding="utf-8"); compile(worker, str(WORKER), "exec")
    nb = nbf.v4.new_notebook(metadata={
        "accelerator":"GPU", "kaggle":{"accelerator":"gpu","dataSources":[]},
        "kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    })
    nb.cells = [
        nbf.v4.new_markdown_cell(f"# Qwen4B original chunks — {scope}, part {part_index+1}/{part_count}\nScores exact submitted chunks; no submission is built on Kaggle."),
        nbf.v4.new_code_cell("%pip install -q --no-cache-dir 'transformers>=4.51,<5' 'accelerate>=1.6,<2' 'huggingface-hub>=0.30,<1' 'pyarrow>=18,<22' 'pandas>=2.2,<3'"),
        code(f'''import hashlib,json,os,subprocess,sys,time,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
SCOPE={scope!r}; PART_INDEX={part_index}; PART_COUNT={part_count}
MODEL_ID={MODEL_ID!r}; REVISION={REVISION!r}; INSTRUCTION={INSTRUCTION!r}
WORK=Path('/kaggle/working/qwen_chunks'); WORK.mkdir(parents=True,exist_ok=True)
try:
 from kaggle_secrets import UserSecretsClient
 token=UserSecretsClient().get_secret('HF_TOKEN')
 if token: os.environ['HF_TOKEN']=token
except Exception: pass
assert torch.cuda.device_count()==2,torch.cuda.device_count()
root=Path('/kaggle/input'); pairs_found=list(root.rglob('qwen_chunk_pairs.parquet')); manifests=list(root.rglob('qwen_chunk_manifest.json'))
if len(pairs_found)!=1 or len(manifests)!=1: raise RuntimeError({{'pairs':pairs_found,'manifests':manifests}})
manifest=json.loads(manifests[0].read_text(encoding='utf-8')); assert manifest['scope']==SCOPE
pairs=pd.read_parquet(pairs_found[0]); qids=sorted(pairs.query_id.astype(int).unique())
owned={{qid for i,qid in enumerate(qids) if i%PART_COUNT==PART_INDEX}}
pairs=pairs[pairs.query_id.isin(owned)].copy().reset_index(drop=True)
# Balance whole queries by estimated tokenization/inference cost. Query-disjoint
# ownership is preserved; only the assignment to the two GPUs changes.
query_cost=pairs.groupby('query_id',sort=True).passage_chars.sum().sort_values(ascending=False,kind='stable')
loads=[0,0]; assigned={{}}
for qid,cost in query_cost.items():
 gpu=0 if loads[0]<=loads[1] else 1; assigned[int(qid)]=gpu; loads[gpu]+=int(cost)
pairs['assigned_gpu']=pairs.query_id.map(assigned).astype('int8')
input_path=WORK/'pairs.parquet'; pairs.to_parquet(input_path,index=False,compression='zstd')
print({{'scope':SCOPE,'part':f'{{PART_INDEX+1}}/{{PART_COUNT}}','queries':len(owned),'pairs':len(pairs),
       'gpu_queries':pairs.groupby('assigned_gpu').query_id.nunique().to_dict(),
       'gpu_pairs':pairs.assigned_gpu.value_counts().sort_index().to_dict(),'estimated_char_load':loads,
       'gpus':[torch.cuda.get_device_name(i) for i in range(2)]}})'''),
        code("WORKER="+repr(worker)+"\nworker_path=WORK/'worker.py'; worker_path.write_text(WORKER,encoding='utf-8'); print({'worker_sha256':hashlib.sha256(WORKER.encode()).hexdigest()})"),
        code('''processes=[]; logs=[]
for gpu in range(2):
 out=WORK/'checkpoints'/f'gpu{gpu}'; out.mkdir(parents=True,exist_ok=True)
 path=WORK/f'gpu{gpu}.log'; log=path.open('w',encoding='utf-8'); logs.append((log,path))
 cmd=[sys.executable,str(worker_path),'--input',str(input_path),'--out',str(out),'--model-id',MODEL_ID,'--revision',REVISION,'--gpu',str(gpu),'--max-length','4096','--instruction',INSTRUCTION]
 env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu); env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
 processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env))
while any(p.poll() is None for p in processes):
 time.sleep(60); state={'returncodes':[p.poll() for p in processes]}
 for _,path in logs:
  if path.exists(): state[path.name]=path.read_text(encoding='utf-8',errors='replace').splitlines()[-1:]
 print(state,flush=True)
for log,_ in logs: log.close()
for process,(_,path) in zip(processes,logs):
 if process.returncode!=0: raise RuntimeError(path.read_text(encoding='utf-8',errors='replace')[-8000:])
frames=[pd.read_parquet(WORK/'checkpoints'/f'gpu{gpu}'/f'qwen4b_gpu{gpu}.parquet') for gpu in range(2)]
scores=pd.concat(frames,ignore_index=True).sort_values(['query_id','chunk_ordinal']).reset_index(drop=True)
assert len(scores)==len(pairs) and scores.pair_id.nunique()==len(pairs)
scores.to_parquet(WORK/'qwen4b_chunk_scores.parquet',index=False,compression='zstd')
validation={'schema_version':'qwen4b_original_chunk_scores_v1','scope':SCOPE,'part_index':PART_INDEX,'part_count':PART_COUNT,'queries':int(scores.query_id.nunique()),'pairs':len(scores),'statuses':scores.status.value_counts().to_dict(),'control_sha256':manifest['control_zip_sha256'],'input_sha256':manifest['artifacts']['qwen_chunk_pairs.parquet']['sha256'],'model_id':MODEL_ID,'revision':REVISION,'max_length':4096}
(WORK/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
(WORK/'run_manifest.json').write_text(json.dumps({'input':manifest,'validation':validation,'instruction':INSTRUCTION},ensure_ascii=False,indent=2),encoding='utf-8')
name=f'qwen4b_chunks_{SCOPE}_part{PART_INDEX+1:02d}of{PART_COUNT:02d}.zip'; target=Path('/kaggle/working')/name
with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
 for name in ['qwen4b_chunk_scores.parquet','validation.json','run_manifest.json']: z.write(WORK/name,arcname=name)
assert zipfile.ZipFile(target).testzip() is None
print(validation); print({'output':str(target),'bytes':target.stat().st_size})''')
    ]
    output.parent.mkdir(parents=True, exist_ok=True); nbf.write(nb, output)
    json.loads(output.read_text(encoding="utf-8"))
    print({"notebook":str(output),"sha256":hashlib.sha256(output.read_bytes()).hexdigest()})


def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--scope",choices=("pilot","full"),required=True)
    p.add_argument("--part-index",type=int,default=0); p.add_argument("--part-count",type=int,default=1)
    p.add_argument("--output",type=Path,required=True); a=p.parse_args()
    if not 0<=a.part_index<a.part_count: raise ValueError("invalid partition")
    generate(a.scope,a.part_index,a.part_count,a.output)


if __name__ == "__main__": main()
