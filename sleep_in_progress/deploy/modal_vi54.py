"""vi54: CPU snapshot to separate weights Volume; offline GPU, retries=0."""
from pathlib import Path
import modal

MODEL = 'nvidia/Nemotron-3-Embed-8B-BF16'
REVISION = 'd1f2f25730bbd775b99b29185134bc86653bf2d1'
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.9.1', 'transformers==5.2.0', 'zstandard==0.23.0')
         .add_local_file(str(Path(__file__).resolve().parents[1]/'scripts/vi54.py'), '/root/vi54.py'))
data = modal.Volume.from_name('r2ai-vi54-data', create_if_missing=True)
weights = modal.Volume.from_name('r2ai-vi54-weights', create_if_missing=True)
app = modal.App('r2ai-vi54', image=image)


@app.function(cpu=2, memory=4096, timeout=3600, retries=0,
              volumes={'/weights': weights, '/data': data})
def download_weights():
    import json
    import time
    from huggingface_hub import snapshot_download
    start=time.monotonic()
    snapshot_download(MODEL, revision=REVISION, local_dir='/weights/model')
    result=dict(model=MODEL, revision=REVISION, seconds=time.monotonic()-start,
                files={str(p.relative_to('/weights/model')):p.stat().st_size
                       for p in Path('/weights/model').rglob('*') if p.is_file()})
    Path('/weights/complete.json').write_text(json.dumps(result))
    weights.commit()
    return result


@app.function(cpu=2, memory=8192, timeout=3600, retries=0,
              volumes={'/weights': weights, '/data': data},
              env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
def ping():
    import json
    import sys
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    sys.path.insert(0,'/root')
    from vi54 import iterz, fingerprint, tokenize
    assert json.loads(Path('/weights/complete.json').read_text())['revision']==REVISION
    config=AutoConfig.from_pretrained('/weights/model', local_files_only=True)
    tok=AutoTokenizer.from_pretrained('/weights/model',padding_side='left',local_files_only=True)
    manifest=json.loads(Path('/data/in/prepare_report.json').read_text())
    assert config.is_causal is False and config.hidden_size==4096
    for name,meta in manifest['files'].items():assert fingerprint(Path('/data/in')/name)==meta
    queries=json.loads(Path('/data/in/queries54.json').read_text())
    qt=tokenize(tok,queries,True)
    assert len(qt)==1200 and {r['id'] for r in qt}==set(range(1,1201))
    parts={};total=0
    for name, meta in manifest['parts'].items():
        path=Path('/data/in')/name
        assert fingerprint(path)=={k:meta[k] for k in ('bytes','sha256')}
        rows=list(iterz(path)); tokenized=tokenize(tok,rows,False)
        tokens=sum(len(r['input_ids']) for r in tokenized);total+=tokens
        parts[name]=dict(documents=len(rows),tokens=tokens,raw=meta)
        print('TOKENS '+name+' '+str(tokens),flush=True)
    result=dict(model=MODEL,revision=REVISION,torch=str(torch.__version__),transformers=str(transformers.__version__),
                architecture=config.model_type,hidden_size=config.hidden_size,padding_side=tok.padding_side,
                tokenizer=type(tok).__name__,parts=parts,tokens=total,documents=sum(p['documents'] for p in parts.values()),
                query_tokens=sum(len(r['input_ids']) for r in qt),all_input_sha=True,is_causal=config.is_causal)
    Path('/data/ping.json').write_text(json.dumps(result));data.commit();return result


@app.function(gpu='H100', cpu=4, memory=65536, timeout=3700, retries=0,
              max_containers=4, scaledown_window=2,
              volumes={'/weights': weights, '/data': data},
              env={'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
def work(mode, job, parts, watchdog):
    import json
    import subprocess
    import sys
    import time
    dest=Path('/data/out')/job;dest.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    try:
        subprocess.run([sys.executable,'/root/vi54.py','gpu','--mode',mode,'--source','/data/in',
                        '--out',str(dest),'--watchdog',str(watchdog),'--parts',*parts],check=True,timeout=watchdog)
        return json.loads((dest/f'{mode}_report.json').read_text())
    finally:
        (dest/'function_wall.json').write_text(json.dumps({'seconds':time.monotonic()-start}));data.commit()
