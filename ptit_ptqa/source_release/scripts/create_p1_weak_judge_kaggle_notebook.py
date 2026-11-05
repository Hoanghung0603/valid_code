"""Generate the compliant 2xT4 P1 weak-judge Kaggle notebook."""

from __future__ import annotations

import hashlib
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/18_p1_weak_judge_14b_2xt4_kaggle.ipynb"


def code(value: str):
    compile(value, "<notebook-cell>", "exec")
    return nbf.v4.new_code_cell(value)


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "kaggle": {"accelerator": "gpu", "dataSources": [], "dockerImageVersionId": None},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    cells = [nbf.v4.new_markdown_cell(
        """# P1 weak relevance judge — Qwen3-14B, 2×T4

Mục đích: tạo **weak labels** cho official-query/source-passage pairs sau pilot reranker. Đây không phải BTC ground truth và không tạo submission. Model open-weight 14B chạy self-host, mỗi T4 một worker 4-bit; output giữ raw response, exact evidence check, model revision, prompt hash và checkpoint để audit.

Input cần có: dataset pilot P1 và output kernel reranker P1. Không dùng `data/dev_set` synthetic cũ."""
    )]
    cells.append(nbf.v4.new_code_cell(
        """%pip install -q --no-deps 'transformers>=4.51,<5' 'huggingface_hub>=0.30,<1' 'accelerate>=1.5,<2' 'bitsandbytes>=0.45,<1' 'safetensors>=0.5,<1'
%pip install -q 'pyarrow>=18,<22' 'pandas>=2.2,<3'"""
    ))
    cells.append(code(
        """import os, sys, json, time, glob, zipfile, hashlib, shutil, subprocess, textwrap, re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from huggingface_hub import snapshot_download

os.environ.setdefault('TOKENIZERS_PARALLELISM','false')
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF','expandable_segments:True')
WORK=Path('/kaggle/working/p1_weak_judge')
WORK.mkdir(parents=True,exist_ok=True)
HF_HOME=Path('/kaggle/working/hf_cache')
HF_HOME.mkdir(parents=True,exist_ok=True)
MODEL_ID='Qwen/Qwen3-14B'
OUTPUT_ZIP=Path('/kaggle/working/p1_weak_judge_outputs.zip')

try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN=UserSecretsClient().get_secret('HF_TOKEN')
except Exception:
    HF_TOKEN=None

def find_one(name):
    matches=glob.glob('/kaggle/input/**/'+name,recursive=True)
    if len(matches)!=1: raise RuntimeError(f'Expected one {name}, found {matches}')
    return Path(matches[0])

def sha256_file(path,block=4*1024*1024):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while b:=f.read(block): h.update(b)
    return h.hexdigest()

assert torch.cuda.device_count()==2, torch.cuda.device_count()
print({'gpus':[torch.cuda.get_device_name(i) for i in range(2)],'disk_free_gb':round(shutil.disk_usage('/kaggle/working').free/1e9,1)})"""
    ))
    cells.append(code(
        r"""reranker_zip=find_one('p1_reranker_pilot_outputs.zip')
pairs_path=find_one('p1_pairs.parquet')
queries_path=find_one('p1_queries.parquet')
with zipfile.ZipFile(reranker_zip) as z:
    assert z.testzip() is None
    doc_scores=pd.read_parquet(z.open('p1_doc_scores.parquet'))
    hard_queue=pd.read_parquet(z.open('p1_annotation_queue.parquet'))
pairs=pd.read_parquet(pairs_path)
queries=pd.read_parquet(queries_path)
assert doc_scores.groupby('query_id').size().min()==216 and doc_scores.groupby('query_id').size().max()==216

# Union: scorer disagreement/hard cases from pilot + deterministic random docs
# + three average-rank anchors.  Reasons are preserved for later bias audit.
selected=[]
for qid,g in doc_scores.groupby('query_id',sort=True):
    reasons={}
    hard_ids=set(hard_queue.loc[hard_queue.query_id==qid,'doc_id'].astype(int))
    for doc in hard_ids: reasons.setdefault(doc,set()).add('hard')
    random_rows=g.sample(n=2,random_state=int(qid)+1701)
    for doc in random_rows.doc_id.astype(int): reasons.setdefault(doc,set()).add('random')
    average_rank=(g.bge_rank+g.qwen_rank)/2
    for anchor in (64,128,192):
        doc=int(g.loc[(average_rank-anchor).abs().idxmin(),'doc_id'])
        reasons.setdefault(doc,set()).add(f'rank_anchor_{anchor}')
    take=g[g.doc_id.isin(reasons)].copy()
    take['selection_reason']=take.doc_id.astype(int).map(lambda doc:'|'.join(sorted(reasons[doc])))
    selected.append(take)
queue=pd.concat(selected,ignore_index=True)

needed=pd.MultiIndex.from_frame(queue[['query_id','doc_id']])
passages=pairs[pd.MultiIndex.from_frame(pairs[['query_id','doc_id']]).isin(needed)].copy()
passages=passages.sort_values(['query_id','doc_id','passage_rank'])
combined=passages.groupby(['query_id','doc_id']).apply(
    lambda g:'\n\n'.join(f'[PASSAGE {int(r.passage_rank)}]\n{r.passage_text}' for r in g.itertuples()),
    include_groups=False).rename('passages').reset_index()
queue=queue.merge(combined,on=['query_id','doc_id'],validate='one_to_one').merge(
    queries[['query_id','query']],on='query_id',validate='many_to_one')
queue=queue.sort_values(['query_id','doc_id']).reset_index(drop=True)
queue['judgment_id']=[f'WJ-{i:05d}' for i in range(1,len(queue)+1)]
queue.to_parquet(WORK/'weak_judge_queue.parquet',index=False,compression='zstd')
print({'rows':len(queue),'queries':queue.query_id.nunique(),'per_query':queue.groupby('query_id').size().describe().to_dict(),'source_zip':sha256_file(reranker_zip)})"""
    ))
    prompt = """Bạn là bộ kiểm định relevance y khoa đa ngôn ngữ. Đánh giá CHỈ dựa trên QUERY và SOURCE PASSAGES.

Nhãn:
- direct: passage chứa bằng chứng trả lời trực tiếp ít nhất một ý quan trọng của query, đúng bệnh/thuốc/population/phủ định/ràng buộc.
- related: cùng chủ đề nhưng không đủ trả lời, quá chung, hoặc lệch một ràng buộc quan trọng.
- irrelevant: không liên quan, sai bệnh/ý hỏi, hoặc chỉ boilerplate.

Không suy diễn kiến thức ngoài passage. Với direct/related, evidence phải là một trích dẫn NGUYÊN VĂN ngắn có trong passages. Trả về đúng một JSON object, không markdown:
{"label":"direct|related|irrelevant","evidence":"exact quote or empty","reason":"brief reason"}"""
    prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    cells.append(code(
        f"""PROMPT={prompt!r}
PROMPT_SHA256={prompt_hash!r}
free_gb=shutil.disk_usage('/kaggle/working').free/1e9
if free_gb < 31:
    raise RuntimeError(f'Need at least ~31 GB free to cache Qwen3-14B weights; found {{free_gb:.1f}} GB')
model_path=snapshot_download(repo_id=MODEL_ID,token=HF_TOKEN,cache_dir=str(HF_HOME))
MODEL_REVISION=Path(model_path).name
config={{
 'model_id':MODEL_ID,'model_path':model_path,'model_revision':MODEL_REVISION,
 'prompt':PROMPT,'prompt_sha256':PROMPT_SHA256,'queue_path':str(WORK/'weak_judge_queue.parquet'),
 'checkpoint_root':str(WORK/'checkpoints'),'max_input_tokens':3584,'max_new_tokens':128,'checkpoint_every':20,
}}
(WORK/'worker_config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
print({{'model':MODEL_ID,'revision':MODEL_REVISION,'prompt_sha256':PROMPT_SHA256}})"""
    ))
    worker_source = r'''import os,sys,json,time,re,hashlib
from pathlib import Path
gpu=int(sys.argv[1]); os.environ['CUDA_VISIBLE_DEVICES']=str(gpu)
os.environ['TOKENIZERS_PARALLELISM']='false'; os.environ['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
import pandas as pd
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM,BitsAndBytesConfig

cfg=json.loads(Path(sys.argv[2]).read_text(encoding='utf-8'))
queue=pd.read_parquet(cfg['queue_path'])
qids=sorted(queue.query_id.unique()); mine=set(qids[gpu::2]); queue=queue[queue.query_id.isin(mine)].reset_index(drop=True)
out_dir=Path(cfg['checkpoint_root'])/f'gpu{gpu}'; out_dir.mkdir(parents=True,exist_ok=True)
out_path=out_dir/f'weak_judgments_gpu{gpu}.parquet'
done={}
if out_path.exists():
    old=pd.read_parquet(out_path)
    done={str(row.judgment_id):row._asdict() for row in old.itertuples(index=False)}

quant=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.float16)
tokenizer=AutoTokenizer.from_pretrained(cfg['model_path'],use_fast=True)
model=AutoModelForCausalLM.from_pretrained(cfg['model_path'],quantization_config=quant,device_map={'':0},torch_dtype=torch.float16,low_cpu_mem_usage=True).eval()

allowed={'direct','related','irrelevant'}
def extract_json(text):
    start=text.find('{'); end=text.rfind('}')
    if start<0 or end<=start: raise ValueError('no_json_object')
    return json.loads(text[start:end+1])
def norm(value): return ' '.join(str(value).split()).casefold()
def save():
    pd.DataFrame(done.values()).sort_values('judgment_id').to_parquet(out_path,index=False,compression='zstd')

started=time.time(); pending=[row for row in queue.itertuples(index=False) if str(row.judgment_id) not in done]
for number,row in enumerate(pending,1):
    user=f"QUERY:\n{row.query}\n\nSOURCE PASSAGES:\n{row.passages}"
    messages=[{'role':'system','content':cfg['prompt']},{'role':'user','content':user}]
    rendered=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    inputs=tokenizer(rendered,return_tensors='pt',truncation=True,max_length=cfg['max_input_tokens']).to('cuda:0')
    input_tokens=int(inputs.input_ids.shape[1]); error=''; parsed={}; raw=''
    try:
        with torch.inference_mode():
            output=model.generate(**inputs,max_new_tokens=cfg['max_new_tokens'],do_sample=False,use_cache=True,pad_token_id=tokenizer.eos_token_id)
        raw=tokenizer.decode(output[0,input_tokens:],skip_special_tokens=True).strip()
        parsed=extract_json(raw)
        label=str(parsed.get('label','')).strip().lower(); evidence=str(parsed.get('evidence','')).strip(); reason=str(parsed.get('reason','')).strip()
        if label not in allowed: raise ValueError('invalid_label')
        evidence_valid=(not evidence and label=='irrelevant') or (bool(evidence) and norm(evidence) in norm(row.passages))
        if label!='irrelevant' and not evidence_valid: error='evidence_not_exact_substring'
    except Exception as exc:
        label=''; evidence=''; reason=''; evidence_valid=False; error=(error+'|'+repr(exc)).strip('|')
    done[str(row.judgment_id)]={
      'judgment_id':str(row.judgment_id),'query_id':int(row.query_id),'doc_id':int(row.doc_id),'gpu':gpu,
      'label':label,'evidence':evidence,'reason':reason,'evidence_valid':bool(evidence_valid),'status':'ok' if label in allowed and not error else 'needs_review',
      'input_tokens':input_tokens,'raw_output':raw,'error':error,'prompt_sha256':cfg['prompt_sha256'],'model_revision':cfg['model_revision'],
    }
    if number%cfg['checkpoint_every']==0 or number==len(pending): save()
    if number%20==0 or number==len(pending):
        elapsed=max(time.time()-started,1e-6); print(json.dumps({'event':'progress','gpu':gpu,'complete':number,'total':len(pending),'rows_per_second':round(number/elapsed,3)}),flush=True)
save(); print(json.dumps({'event':'complete','gpu':gpu,'rows':len(done)}),flush=True)
'''
    compile(worker_source, "<weak-judge-worker>", "exec")
    cells.append(code(
        f"""WORKER_SOURCE={worker_source!r}
worker_path=WORK/'weak_judge_worker.py'
worker_path.write_text(WORKER_SOURCE,encoding='utf-8')
logs=[]; processes=[]
for gpu in range(2):
    log_path=WORK/f'worker_gpu{{gpu}}.log'; handle=open(log_path,'w',encoding='utf-8'); logs.append((handle,log_path))
    processes.append(subprocess.Popen([sys.executable,str(worker_path),str(gpu),str(WORK/'worker_config.json')],stdout=handle,stderr=subprocess.STDOUT,text=True))
    if gpu==0: time.sleep(90)  # avoid simultaneous 14B CPU/disk load spikes
while any(p.poll() is None for p in processes):
    status=[]
    for gpu,(process,(_,path)) in enumerate(zip(processes,logs)):
        lines=path.read_text(encoding='utf-8',errors='replace').splitlines() if path.exists() else []
        status.append({{'gpu':gpu,'returncode':process.poll(),'tail':lines[-1:]}})
    print(status,flush=True); time.sleep(60)
for handle,_ in logs: handle.close()
for process,(_,path) in zip(processes,logs):
    if process.returncode!=0: raise RuntimeError(f'worker failed {{process.returncode}}:\\n'+path.read_text(encoding='utf-8',errors='replace')[-8000:])"""
    ))
    cells.append(code(
        """frames=[pd.read_parquet(WORK/'checkpoints'/f'gpu{gpu}'/f'weak_judgments_gpu{gpu}.parquet') for gpu in range(2)]
judgments=pd.concat(frames,ignore_index=True).sort_values('judgment_id').reset_index(drop=True)
assert len(judgments)==len(queue) and judgments.judgment_id.nunique()==len(queue)
judgments=queue.merge(judgments,on=['judgment_id','query_id','doc_id'],validate='one_to_one')
judgments.to_parquet(WORK/'p1_weak_judgments.parquet',index=False,compression='zstd')
audit={
 'schema_version':'p1_weak_judge_14b_v1','model_id':MODEL_ID,'model_revision':MODEL_REVISION,'prompt_sha256':PROMPT_SHA256,
 'rows':len(judgments),'queries':judgments.query_id.nunique(),'status':judgments.status.value_counts().to_dict(),
 'labels':judgments.label.value_counts(dropna=False).to_dict(),'exact_evidence_rate':float(judgments.evidence_valid.mean()),
 'source_reranker_zip_sha256':sha256_file(reranker_zip),'source_pairs_sha256':sha256_file(pairs_path),
 'warning':'Weak labels are not BTC ground truth; human review is required for disagreements and calibration holdout.',
}
(WORK/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf-8')
members=['p1_weak_judgments.parquet','audit.json']
with zipfile.ZipFile(OUTPUT_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
    for name in members: z.write(WORK/name,arcname=name)
assert zipfile.ZipFile(OUTPUT_ZIP).testzip() is None
print(audit)
print({'output':str(OUTPUT_ZIP),'bytes':OUTPUT_ZIP.stat().st_size,'sha256':sha256_file(OUTPUT_ZIP)})"""
    ))
    notebook["cells"] = cells
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    print({"notebook": str(OUTPUT), "cells": len(cells), "sha256": hashlib.sha256(OUTPUT.read_bytes()).hexdigest()})


if __name__ == "__main__":
    main()
