"""vi50 uses identical vi45 model/image/prompt; two H100 maximum per workspace."""
from pathlib import Path
import modal

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('vllm==0.15.1','zstandard==0.23.0')
    .env({'HF_HOME':'/hf'})
    .run_commands('python -c "from huggingface_hub import snapshot_download; snapshot_download(\'Qwen/Qwen3-Reranker-8B\')"')
    .add_local_dir(str(SCRIPTS),'/root/scripts'))
vol=modal.Volume.from_name('r2ai-runner',create_if_missing=False)
app=modal.App('r2ai-vi50',image=image)


@app.function(cpu=1,memory=2048,timeout=120,retries=0,volumes={'/vol':vol})
def ping():
    import importlib
    import json
    import sys
    sys.path.insert(0,'/root/scripts')
    result={n:str(getattr(importlib.import_module(n),'__version__','ok'))
        for n in ('vllm','torch','transformers','vi45','qwen_rerank_run')}
    dest=Path('/vol/out/vi50');dest.mkdir(parents=True,exist_ok=True)
    (dest/'ping.json').write_text(json.dumps(result))
    vol.commit()
    return result


@app.function(gpu='H100',cpu=4,memory=49152,timeout=2500,retries=0,
    max_containers=2,scaledown_window=2,volumes={'/vol':vol})
def bulk(job,watchdog):
    import json
    import subprocess
    import sys
    import time
    assert watchdog<=2400
    vol.reload()
    assert Path('/vol/out/vi50/ping.json').exists()
    src=Path('/vol/in/vi50/jobs')/job
    dest=Path('/vol/out/vi50/bulk')/job
    dest.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    try:
        subprocess.run([sys.executable,'/root/scripts/vi45.py','gpu','--mode','bulk',
            '--input',str(src),'--output',str(dest)],check=True,timeout=watchdog)
        return json.loads((dest/'bulk_report.json').read_text())
    finally:
        (dest/'function_wall.json').write_text(json.dumps({'seconds':time.monotonic()-start,'hourly':4.5232}))
        vol.commit()
