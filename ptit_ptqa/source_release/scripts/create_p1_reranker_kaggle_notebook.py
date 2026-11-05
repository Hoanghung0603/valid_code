"""Generate the self-contained 2xT4 Kaggle notebook for P1 reranker pilot."""

from __future__ import annotations

import json
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks/17_p1_reranker_pilot_2xt4_kaggle.ipynb"


def main() -> None:
    notebook = nbf.v4.new_notebook()
    notebook["metadata"] = {
        "accelerator": "GPU",
        "kaggle": {"accelerator": "gpu", "dataSources": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    cells = []
    cells.append(nbf.v4.new_markdown_cell(
        """# R2AI P1 — evidence-aware reranker pilot (2×T4)

Scores the same 40k query–passage pairs with BGE reranker v2 m3 and
Qwen3-Reranker-4B.  Two independent workers split query IDs across the two
T4s.  The notebook never creates a competition submission and never changes
the candidate pool; it only emits auditable scores for local calibration.

Required private dataset: `ptquanh/r2ai-p1-reranker-pilot-v1`.  Optional
Kaggle Secret: `HF_TOKEN`."""
    ))
    cells.append(nbf.v4.new_code_cell(
        """%pip install -q --no-cache-dir \"transformers>=4.51,<5\" \"accelerate>=1.6,<2\" \"huggingface-hub>=0.30,<1\" \"pyarrow>=18,<22\" \"pandas>=2.2,<3\""""
    ))
    cells.append(nbf.v4.new_code_cell(
        """import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch

WORK = Path('/kaggle/working')
INPUT_ROOT = Path('/kaggle/input')
OUTPUT_ZIP = WORK / 'p1_reranker_pilot_outputs.zip'
INSTRUCTION = 'Given a multilingual medical question, retrieve passages that contain information needed to answer the question while respecting clinical intent, population, treatment, and negation constraints.'

try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret('HF_TOKEN')
    if token:
        os.environ['HF_TOKEN'] = token
except Exception:
    pass

assert torch.cuda.device_count() == 2, f'Expected Kaggle GPU T4 x2, got {torch.cuda.device_count()} GPU(s)'
PACK_DIR = WORK / 'p1_input'
if PACK_DIR.exists(): shutil.rmtree(PACK_DIR)
PACK_DIR.mkdir(parents=True)
zip_candidates = list(INPUT_ROOT.rglob('p1_reranker_pilot_v1.zip'))
pair_candidates = list(INPUT_ROOT.rglob('p1_pairs.parquet'))
if len(zip_candidates) == 1:
    with zipfile.ZipFile(zip_candidates[0]) as archive:
        assert archive.testzip() is None
        archive.extractall(PACK_DIR)
elif len(pair_candidates) == 1:
    # Kaggle Datasets automatically expands an uploaded ZIP into member files.
    source_dir = pair_candidates[0].parent
    for name in ('p1_pairs.parquet','p1_queries.parquet','p1_manifest.json'):
        shutil.copy2(source_dir/name, PACK_DIR/name)
else:
    raise RuntimeError(f'Expected one packed or expanded P1 dataset; zips={zip_candidates}, pairs={pair_candidates}')

manifest = json.loads((PACK_DIR/'p1_manifest.json').read_text(encoding='utf-8'))
pairs = pd.read_parquet(PACK_DIR/'p1_pairs.parquet')
queries = pd.read_parquet(PACK_DIR/'p1_queries.parquet')
assert len(queries) == 120 and len(pairs) == manifest['validation']['pairs']
assert pairs.pair_id.nunique() == len(pairs)
assert pairs.groupby(['query_id','doc_id']).ngroups == 120*216
print({'gpus': [torch.cuda.get_device_name(i) for i in range(2)], 'pairs': len(pairs), 'queries': len(queries), 'input_sha256': manifest['artifacts']['p1_pairs.parquet']['sha256']})"""
    ))
    worker = r'''from __future__ import annotations
import argparse, gc, json, math, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--input',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--model-key',choices=['bge','qwen4b'],required=True); p.add_argument('--model-id',required=True)
    p.add_argument('--gpu',type=int,required=True); p.add_argument('--world-size',type=int,default=2)
    p.add_argument('--batch-size',type=int,required=True); p.add_argument('--max-length',type=int,default=1024)
    p.add_argument('--instruction',required=True); return p.parse_args()

def write_part(out_dir, number, rows):
    if not rows: return
    pd.DataFrame.from_records(rows).to_parquet(out_dir/f'part_{number:06d}.parquet',index=False,compression='zstd')

def load_done(out_dir):
    done=set()
    for p in sorted(out_dir.glob('part_*.parquet')):
        done.update(pd.read_parquet(p,columns=['pair_id']).pair_id.astype(str))
    return done

def main():
    a=parse_args(); torch.cuda.set_device(0); device=torch.device('cuda:0')
    frame=pd.read_parquet(a.input).reset_index(drop=True)
    qids=sorted(frame.query_id.astype(int).unique())
    owned={qid for i,qid in enumerate(qids) if i%a.world_size==a.gpu}
    frame=frame[frame.query_id.isin(owned)].copy().reset_index(drop=True)
    a.out.mkdir(parents=True,exist_ok=True); done=load_done(a.out)
    pending=frame[~frame.pair_id.astype(str).isin(done)].copy().reset_index(drop=True)
    print(json.dumps({'event':'start','model':a.model_key,'gpu':a.gpu,'rows':len(frame),'pending':len(pending)}),flush=True)
    tokenizer=AutoTokenizer.from_pretrained(a.model_id,padding_side='left' if a.model_key=='qwen4b' else 'right')
    revision='unknown'; prefix_tokens=suffix_tokens=None; yes_id=no_id=None
    if a.model_key=='bge':
        model=AutoModelForSequenceClassification.from_pretrained(a.model_id,torch_dtype=torch.float16).to(device).eval()
    else:
        model=AutoModelForCausalLM.from_pretrained(a.model_id,torch_dtype=torch.float16,attn_implementation='sdpa').to(device).eval()
        no_id=tokenizer.convert_tokens_to_ids('no'); yes_id=tokenizer.convert_tokens_to_ids('yes')
        prefix='<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
        suffix='<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
        prefix_tokens=tokenizer.encode(prefix,add_special_tokens=False); suffix_tokens=tokenizer.encode(suffix,add_special_tokens=False)
        if tokenizer.pad_token_id is None: tokenizer.pad_token=tokenizer.eos_token
    revision=getattr(model.config,'_commit_hash',None) or 'unknown'
    started=time.time(); buffer=[]; part_no=len(list(a.out.glob('part_*.parquet')))
    with torch.inference_mode():
        for start in range(0,len(pending),a.batch_size):
            block=pending.iloc[start:start+a.batch_size]
            queries=block.query_text.astype(str).tolist(); docs=block.passage_text.astype(str).tolist()
            if a.model_key=='bge':
                inputs=tokenizer(queries,docs,padding=True,truncation='longest_first',max_length=a.max_length,return_tensors='pt')
                token_counts=inputs['attention_mask'].sum(1).tolist(); inputs={k:v.to(device) for k,v in inputs.items()}
                raw=model(**inputs).logits.float().reshape(-1); norm=torch.sigmoid(raw)
            else:
                formatted=[f'<Instruct>: {a.instruction}\n<Query>: {q}\n<Document>: {d}' for q,d in zip(queries,docs)]
                encoded=tokenizer(formatted,padding=False,truncation='longest_first',max_length=a.max_length-len(prefix_tokens)-len(suffix_tokens))
                ids=[prefix_tokens+x+suffix_tokens for x in encoded['input_ids']]
                inputs=tokenizer.pad({'input_ids':ids},padding=True,return_attention_mask=True,return_tensors='pt')
                token_counts=inputs['attention_mask'].sum(1).tolist(); inputs={k:v.to(device) for k,v in inputs.items()}
                final=model(**inputs).logits[:,-1,:].float(); raw=final[:,yes_id]-final[:,no_id]; norm=torch.sigmoid(raw)
            for row,raw_score,norm_score,tokens in zip(block.itertuples(index=False),raw.cpu().tolist(),norm.cpu().tolist(),token_counts):
                buffer.append({'pair_id':str(row.pair_id),'query_id':int(row.query_id),'doc_id':int(row.doc_id),'passage_rank':int(row.passage_rank),'raw_score':float(raw_score),'normalized_score':float(norm_score),'input_tokens':int(tokens),'model_key':a.model_key,'model_id':a.model_id,'model_revision':revision,'gpu_shard':a.gpu})
            if len(buffer)>=512 or start+a.batch_size>=len(pending):
                write_part(a.out,part_no,buffer); part_no+=1; buffer=[]
            if (start//a.batch_size+1)%50==0 or start+a.batch_size>=len(pending):
                elapsed=time.time()-started; complete=min(start+a.batch_size,len(pending)); print(json.dumps({'event':'progress','model':a.model_key,'gpu':a.gpu,'complete':complete,'total':len(pending),'pairs_per_second':round(complete/max(elapsed,1e-6),3)}),flush=True)
    del model; gc.collect(); torch.cuda.empty_cache()
    all_parts=[pd.read_parquet(p) for p in sorted(a.out.glob('part_*.parquet'))]
    scores=pd.concat(all_parts,ignore_index=True).drop_duplicates('pair_id',keep='last')
    expected=set(frame.pair_id.astype(str)); actual=set(scores.pair_id.astype(str)); assert actual==expected,(len(expected-actual),len(actual-expected))
    scores.sort_values(['query_id','doc_id','passage_rank']).to_parquet(a.out/f'{a.model_key}_gpu{a.gpu}.parquet',index=False,compression='zstd')
    (a.out/f'{a.model_key}_gpu{a.gpu}.json').write_text(json.dumps({'model_key':a.model_key,'model_id':a.model_id,'model_revision':revision,'gpu':a.gpu,'rows':len(scores),'seconds':time.time()-started},indent=2),encoding='utf-8')
    print(json.dumps({'event':'complete','model':a.model_key,'gpu':a.gpu,'rows':len(scores)}),flush=True)
if __name__=='__main__': main()
'''
    cells.append(nbf.v4.new_code_cell(
        "WORKER = " + repr(worker) + "\n(WORK/'p1_worker.py').write_text(WORKER, encoding='utf-8')\nprint('Worker source written:', len(WORKER), 'chars')"
    ))
    cells.append(nbf.v4.new_code_cell(
        """MODELS = {
    'bge': {'id':'BAAI/bge-reranker-v2-m3','batch':16},
    'qwen4b': {'id':'Qwen/Qwen3-Reranker-4B','batch':4},
}

def run_model(model_key):
    spec=MODELS[model_key]; processes=[]; logs=[]
    for gpu in range(2):
        out=WORK/'checkpoints'/model_key/f'gpu{gpu}'; out.mkdir(parents=True,exist_ok=True)
        log_path=WORK/f'{model_key}_gpu{gpu}.log'; log=log_path.open('w',encoding='utf-8'); logs.append((log,log_path))
        cmd=[sys.executable,str(WORK/'p1_worker.py'),'--input',str(PACK_DIR/'p1_pairs.parquet'),'--out',str(out),'--model-key',model_key,'--model-id',spec['id'],'--gpu',str(gpu),'--batch-size',str(spec['batch']),'--max-length','1024','--instruction',INSTRUCTION]
        env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu); env['PYTORCH_CUDA_ALLOC_CONF']='expandable_segments:True'
        processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env))
    while any(p.poll() is None for p in processes):
        time.sleep(60)
        status={'model':model_key,'returncodes':[p.poll() for p in processes]}
        for _,path in logs:
            if path.exists(): status[path.name]=path.read_text(encoding='utf-8',errors='replace').splitlines()[-1:]
        print(status,flush=True)
    for log,_ in logs: log.close()
    for p,(_,path) in zip(processes,logs):
        if p.returncode!=0: raise RuntimeError(f'{model_key} worker failed ({p.returncode}):\\n'+path.read_text(encoding='utf-8',errors='replace')[-6000:])
    frames=[]
    for gpu in range(2): frames.append(pd.read_parquet(WORK/'checkpoints'/model_key/f'gpu{gpu}'/f'{model_key}_gpu{gpu}.parquet'))
    result=pd.concat(frames,ignore_index=True).sort_values(['query_id','doc_id','passage_rank']).reset_index(drop=True)
    assert len(result)==len(pairs) and result.pair_id.nunique()==len(pairs) and np.isfinite(result.raw_score).all()
    result.to_parquet(WORK/f'{model_key}_scores.parquet',index=False,compression='zstd')
    print({'model':model_key,'rows':len(result),'raw_min':result.raw_score.min(),'raw_mean':result.raw_score.mean(),'raw_max':result.raw_score.max()})
    return result

bge_scores=run_model('bge')"""
    ))
    cells.append(nbf.v4.new_code_cell("qwen_scores=run_model('qwen4b')"))
    cells.append(nbf.v4.new_code_cell(
        """base_cols=['pair_id','query_id','doc_id','doc_rank','owner_doc','passage_rank','chunk_id','language','dense_score','start','end','source_sha256','passage_sha256']
scored=pairs[base_cols].merge(bge_scores[['pair_id','raw_score','normalized_score','input_tokens']].rename(columns={'raw_score':'bge_raw','normalized_score':'bge_score','input_tokens':'bge_tokens'}),on='pair_id',validate='one_to_one')
scored=scored.merge(qwen_scores[['pair_id','raw_score','normalized_score','input_tokens']].rename(columns={'raw_score':'qwen_raw','normalized_score':'qwen_score','input_tokens':'qwen_tokens'}),on='pair_id',validate='one_to_one')
scored.to_parquet(WORK/'p1_pair_scores.parquet',index=False,compression='zstd')

docs=scored.groupby(['query_id','doc_id','doc_rank','owner_doc'],as_index=False).agg(
    bge_score=('bge_score','max'),qwen_score=('qwen_score','max'),
    bge_raw=('bge_raw','max'),qwen_raw=('qwen_raw','max'),
    passage_count=('pair_id','size'),language=('language','first'))
docs['bge_rank']=docs.groupby('query_id').bge_score.rank(method='first',ascending=False).astype(int)
docs['qwen_rank']=docs.groupby('query_id').qwen_score.rank(method='first',ascending=False).astype(int)
docs['rank_disagreement']=(docs.bge_rank-docs.qwen_rank).abs()
docs.to_parquet(WORK/'p1_doc_scores.parquet',index=False,compression='zstd')

queue=[]
for qid,group in docs.groupby('query_id',sort=True):
    candidates=pd.concat([
        group[group.owner_doc].nsmallest(2,'qwen_score'),
        group[~group.owner_doc].nlargest(2,'qwen_score'),
        group.nlargest(2,'rank_disagreement'),
    ]).drop_duplicates(['query_id','doc_id'])
    queue.append(candidates)
annotation=pd.concat(queue,ignore_index=True).merge(queries[['query_id','query']],on='query_id',validate='many_to_one')
annotation['label']=''; annotation['notes']=''
annotation.to_parquet(WORK/'p1_annotation_queue.parquet',index=False,compression='zstd')

agreement=docs.groupby('query_id').apply(lambda g:g.bge_score.rank().corr(g.qwen_score.rank()),include_groups=False)
validation={
 'input_pairs':len(pairs),'pair_scores':len(scored),'doc_scores':len(docs),'queries':docs.query_id.nunique(),
 'query_docs_per_query_min':int(docs.groupby('query_id').size().min()),'query_docs_per_query_max':int(docs.groupby('query_id').size().max()),
 'bge_nonfinite':int((~np.isfinite(scored.bge_raw)).sum()),'qwen_nonfinite':int((~np.isfinite(scored.qwen_raw)).sum()),
 'mean_spearman_between_models':float(agreement.mean()),'annotation_rows':len(annotation),
 'control_sha256':manifest['control_zip_sha256'],'input_pairs_sha256':manifest['artifacts']['p1_pairs.parquet']['sha256'],
}
(WORK/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
run_manifest={'schema_version':'p1_reranker_pilot_outputs_v1','instruction':INSTRUCTION,'input_manifest':manifest,'models':MODELS,'validation':validation}
(WORK/'run_manifest.json').write_text(json.dumps(run_manifest,ensure_ascii=False,indent=2),encoding='utf-8')

members=['bge_scores.parquet','qwen4b_scores.parquet','p1_pair_scores.parquet','p1_doc_scores.parquet','p1_annotation_queue.parquet','validation.json','run_manifest.json']
with zipfile.ZipFile(OUTPUT_ZIP,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as archive:
    for name in members: archive.write(WORK/name,arcname=name)
assert zipfile.ZipFile(OUTPUT_ZIP).testzip() is None
print(validation)
print({'output':str(OUTPUT_ZIP),'bytes':OUTPUT_ZIP.stat().st_size,'sha256':hashlib.sha256(OUTPUT_ZIP.read_bytes()).hexdigest()})"""
    ))
    notebook["cells"] = cells
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(notebook, OUTPUT)
    # Parse the generated JSON again; a malformed notebook must never be pushed.
    json.loads(OUTPUT.read_text(encoding="utf-8"))
    print(OUTPUT)


if __name__ == "__main__":
    main()
