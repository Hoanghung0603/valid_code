"""Generate self-contained 2xT4 notebooks with exact-input scoring and private metadata."""
import argparse
import json
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--scope', choices=['pilot', 'full'], required=True)
    args = p.parse_args()
    scope = args.scope
    worker = (ROOT/'scripts/p2b_score_worker.py').read_text(encoding='utf-8')
    cells = [nbf.v4.new_markdown_cell(f'''# P2b paragraph localization — {scope}
Original VI query only. No labels/local evaluation. Exact proposed spans are scored without
truncation; oversize inputs are marked `too_long`, never scored on a hidden prefix.
Pinned BGE revision; two independent GPU workers. Optional Kaggle secret `HF_TOKEN`.
Download only `p2b_{scope}_scores.zip`. Pilot is a structural/runtime check, not a quality gate.'''),
    nbf.v4.new_code_cell("%pip install -q 'transformers>=4.51,<5' 'accelerate>=1.6,<2' 'huggingface-hub>=0.30,<1' 'pyarrow>=18,<22' 'pandas>=2.2,<3'")]
    source = f'''import os, json, hashlib, subprocess, sys, time, zipfile
from pathlib import Path
import pandas as pd
import torch
SCOPE={scope!r}
WORK=Path('/kaggle/temp/p2b_'+SCOPE); WORK.mkdir(parents=True,exist_ok=True)
os.environ['HF_HOME']='/kaggle/temp/p2b_hf_cache'
try:
    from kaggle_secrets import UserSecretsClient
    token=UserSecretsClient().get_secret('HF_TOKEN')
    if token: os.environ['HF_TOKEN']=token
except Exception:
    pass
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        while block:=f.read(4*1024*1024): h.update(block)
    return h.hexdigest()
assert torch.cuda.device_count()==2
paths=list(Path('/kaggle/input').rglob('p2b_manifest.json'))
assert len(paths)==1,paths
manifest=json.loads(paths[0].read_text())
pair_path=paths[0].parent/'p2b_pairs.parquet'
assert manifest['scope']==SCOPE and manifest['schema']=='p2b_localization_v1'
assert sha(pair_path)==manifest['pair_sha256']
print(manifest,flush=True)
'''
    cells.append(nbf.v4.new_code_cell(source))
    cells.append(nbf.v4.new_code_cell('WORKER='+repr(worker)+"\nworker_path=WORK/'worker.py'\nworker_path.write_text(WORKER,encoding='utf-8')"))
    cells.append(nbf.v4.new_code_cell('''processes=[]; logs=[]
for gpu in range(2):
    log_path=WORK/f'gpu{gpu}.log'; log=log_path.open('w'); logs.append((log,log_path))
    env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=str(gpu)
    cmd=[sys.executable,str(worker_path),'--input',str(pair_path),'--output',str(WORK/f'gpu{gpu}'),'--shard',str(gpu)]
    processes.append(subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env))
while any(p.poll() is None for p in processes):
    time.sleep(30)
    print({'returncodes':[p.poll() for p in processes],
        'tail':[path.read_text(errors='replace').splitlines()[-1:] for _,path in logs]},flush=True)
for log,_ in logs: log.close()
for p,(_,path) in zip(processes,logs):
    if p.returncode: raise RuntimeError(path.read_text(errors='replace')[-5000:])
scores=pd.concat([pd.read_parquet(WORK/f'gpu{gpu}'/'scores.parquet') for gpu in range(2)],ignore_index=True)
ids=pd.read_parquet(pair_path,columns=['pair_id']).pair_id
assert not scores.pair_id.duplicated().any() and set(scores.pair_id)==set(ids)
assert set(scores.revision)=={manifest['revision']}
out=WORK/'scores.parquet'; scores.to_parquet(out,index=False)
run={'input':manifest,'score_sha256':sha(out),'statuses':scores.status.value_counts().to_dict(),
     'torch':torch.__version__,'truncation':False,'worker_sha256':sha(worker_path),
     'workers':[json.loads((WORK/f'gpu{gpu}'/'identity.json').read_text()) for gpu in range(2)]}
(WORK/'run.json').write_text(json.dumps(run,indent=2))
target=Path('/kaggle/working')/f'p2b_{SCOPE}_scores.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
    z.write(out,'scores.parquet'); z.write(WORK/'run.json','run.json')
print({'output':str(target),'sha256':sha(target),'statuses':run['statuses']})
'''))
    for cell in cells[2:]:
        compile(cell.source, '<notebook>', 'exec')
    nb = nbf.v4.new_notebook(cells=cells, metadata=dict(kernelspec=dict(
        display_name='Python 3', language='python', name='python3')))
    name = f'21_p2b_localization_{scope}_2xt4_kaggle.ipynb'
    nbf.write(nb, ROOT/'notebooks'/name)
    upload = ROOT/f'data/kaggle_inputs/p2b_{scope}_kernel'
    upload.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, upload/name)
    meta = dict(id=f'ptquanh/r2ai-p2b-localization-{scope}', title=f'R2AI P2b Localization {scope}',
                code_file=name, language='python', kernel_type='notebook', is_private='true',
                enable_gpu='true', enable_tpu='false', enable_internet='true', machine_shape='NvidiaTeslaT4',
                dataset_sources=[f'ptquanh/r2ai-p2b-localization-{scope}-v1'],
                competition_sources=[], kernel_sources=[], model_sources=[])
    (upload/'kernel-metadata.json').write_text(json.dumps(meta, indent=2))
    print(upload)


if __name__ == '__main__':
    main()
