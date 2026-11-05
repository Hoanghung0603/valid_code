"""Reuse runner score implementations, with isolated paths and hard subprocess deadlines."""
import modal
from pathlib import Path
SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.6.0', 'transformers==4.51.3', 'zstandard==0.23.0',
                 'numpy==1.26.4', 'huggingface_hub==0.30.2')
    .env({'HF_HOME':'/hf', 'HF_HUB_CACHE':'/hf'})
    .run_commands("python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-Reranker-8B')\"")
    .run_commands("python -c \"from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-reranker-v2-m3')\"")
    .add_local_dir(str(SCRIPTS), '/root/scripts'))

vol=modal.Volume.from_name('r2ai-runner',create_if_missing=False)
app=modal.App('r2ai-vi41-score',image=image)


def guarded(job, function):
    import json
    flag=Path('/vol/out/vi41/score_init_failed.json')
    assert not flag.exists(), 'VI41 INIT GUARD: '+str(flag)
    try:
        return function()
    except Exception as exc:
        flag.parent.mkdir(parents=True,exist_ok=True)
        flag.write_text(json.dumps({'job':job,'error':repr(exc)}))
        vol.commit()
        raise


@app.function(cpu=1,memory=2048,timeout=120,retries=0,volumes={'/vol':vol})
def ping():
    def run():
        import importlib
        import json
        import sys
        sys.path.insert(0,'/root/scripts')
        names=('torch','transformers','numpy','zstandard','rerank_run','docwin_run',
               'qwen_rerank_run','dwmax_diag')
        versions={n:str(getattr(importlib.import_module(n),'__version__','import-ok')) for n in names}
        Path('/vol/out/vi41/score_ping.json').write_text(json.dumps(versions))
        vol.commit()
        return versions
    return guarded('ping',run)


@app.function(gpu='L40S',cpu=4,memory=32768,timeout=11000,
              scaledown_window=2,max_containers=1,retries=0,volumes={'/vol':vol})
def score(stage: str, seconds: int):
    assert Path('/vol/out/vi41/score_ping.json').exists(), 'CPU ping required'
    return guarded(stage,lambda: run_score(stage,seconds))


def run_score(stage: str, seconds: int):
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path
    assert stage in ('rr','docwin','q8b') and 60<=seconds<=10500
    start=time.monotonic()
    src,dest=Path('/vol/in/vi41'),Path('/vol/out/vi41')
    dest.mkdir(parents=True,exist_ok=True)
    docs=sorted(str(p) for p in src.glob('docs_new.part*.jsonl.zst'))
    assert docs
    script={'rr':'rerank_run.py','docwin':'docwin_run.py','q8b':'qwen_rerank_run.py'}[stage]
    output={'rr':'rr_new.json','docwin':'docwin_new.json.zst','q8b':'q8b_new.json'}[stage]
    command=[sys.executable,'/root/scripts/'+script,'--docs',*docs,'--queries',str(src/'queries.jsonl'),
        '--cands',str(src/('q8b_cands.json' if stage=='q8b' else 'new_cands.json')),
        '--out',str(dest/output),'--resume-dir',str(dest/('resume_'+stage))]
    if stage=='rr':
        command+=['--max-len','512','--batch','16','--top','800']
    elif stage=='docwin':
        command+=['--max-len','512','--batch','16','--top-docs','500','--window-words','300','--window-stride','250']
    else:
        command+=['--model','Qwen/Qwen3-Reranker-8B','--dtype','fp16','--max-len','1024','--batch','8',
                  '--top','1000','--no-watchdog']
    report=dict(stage=stage,seconds_budget=seconds)
    try:
        subprocess.run(command,check=True,timeout=seconds)
        report['completed']=True
        return report
    finally:
        report['seconds']=time.monotonic()-start
        (dest/(stage+'_function_report.json')).write_text(json.dumps(report,indent=2))
        vol.commit()


@app.function(gpu='L40S',cpu=4,memory=32768,timeout=300,
              scaledown_window=2,max_containers=1,retries=0,volumes={'/vol':vol})
def bench():
    assert Path('/vol/out/vi41/score_ping.json').exists(), 'CPU ping required'
    return guarded('bench',run_bench)


def run_bench():
    import json
    import sys
    import time
    sys.path.insert(0,'/root/scripts')
    from pathlib import Path
    from rerank_run import load_docs,load_model,score_pairs
    import qwen_rerank_run as qrr
    import torch
    start=time.monotonic()
    src=Path('/vol/in/vi37')
    dest=Path('/vol/out/vi41')
    dest.mkdir(parents=True,exist_ok=True)
    rows=json.loads((src/'bench_docwin_in.json').read_text())
    tok,model=load_model('BAAI/bge-reranker-v2-m3')
    s=score_pairs(model,tok,[(r['query'],r['text']) for r in rows],512,16)
    pairs=[(r['score'],float(v)) for r,v in zip(rows,s)]
    del model
    torch.cuda.empty_cache()
    from dwmax_diag import overlap_report
    bge=overlap_report(pairs)
    assert bge['pearson']>=.999,bge
    ref=json.loads((src/'bench_q8b_old.json').read_text())
    wanted={int(d) for q,d,v in ref}
    docs=load_docs([str(p) for p in src.glob('docs_q8b.part*.jsonl.zst')],wanted=wanted)
    queries={str(r['id']):r['query'] for r in [json.loads(l) for l in (src/'queries.jsonl').read_text().splitlines()]}
    tok,model,yes,no=qrr.load_model('Qwen/Qwen3-Reranker-8B','fp16')
    scores,stats=qrr.score_pairs(model,tok,[(queries[q],docs[int(d)]) for q,d,v in ref],
                                qrr.DEFAULT_INSTRUCTION,1024,8,yes,no)
    q8=overlap_report([(v,float(s)) for (q,d,v),s in zip(ref,scores)])
    assert q8['pearson']>=.999,q8
    report=dict(bge=bge,q8b=q8,seconds=time.monotonic()-start)
    (dest/'bench_report.json').write_text(json.dumps(report,indent=2))
    vol.commit()
    return report
