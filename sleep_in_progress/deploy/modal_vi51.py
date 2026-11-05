"""vi51 isolated image/Volume, explicit deployed Function.spawn, retries=0."""
from pathlib import Path
import os
import modal

SCRIPTS=Path(__file__).resolve().parents[1]/'scripts'
image=(modal.Image.debian_slim(python_version='3.11')
    .pip_install('torch==2.9.1','transformers==4.57.6','zstandard==0.23.0')
    .env({'HF_HOME':'/hf'})
    .run_commands('python -c "from huggingface_hub import snapshot_download; snapshot_download(\'Qwen/Qwen3-Embedding-0.6B\',revision=\'97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3\'); snapshot_download(\'Qwen/Qwen3-Embedding-8B\',revision=\'1d8ad4ca9b3dd8059ad90a75d4983776a23d44af\')"')
    .add_local_dir(str(SCRIPTS),'/root/scripts'))
vol=modal.Volume.from_name('r2ai-vi51-data',create_if_missing=True)
app=modal.App('r2ai-vi51',image=image)


@app.function(cpu=1,memory=2048,timeout=120,retries=0,volumes={'/vol':vol})
def ping():
    import importlib,json,sys
    sys.path.insert(0,'/root/scripts')
    versions={n:str(getattr(importlib.import_module(n),'__version__','ok')) for n in ('torch','transformers','vi51')}
    from transformers import AutoModel,AutoTokenizer
    dest=Path('/vol/out');dest.mkdir(exist_ok=True)
    (dest/'ping.json').write_text(json.dumps(versions));vol.commit();return versions


@app.function(gpu='H100',cpu=4,memory=65536,timeout=3700,retries=0,
    max_containers=int(os.environ.get('VI51_MAX_CONTAINERS','4')),scaledown_window=2,volumes={'/vol':vol})
def work(tier,mode,job,parts,watchdog):
    import json,subprocess,sys,time
    vol.reload();assert Path('/vol/out/ping.json').exists()
    assert 1<=watchdog<=3600
    dest=Path('/vol/out')/tier/job;dest.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    try:
        subprocess.run([sys.executable,'/root/scripts/vi51.py','gpu','--tier',tier,'--mode',mode,
            '--source','/vol/in','--out',str(dest),'--watchdog',str(watchdog),'--parts',*parts],
            check=True,timeout=watchdog)
        return json.loads((dest/('bench_report.json' if mode=='bench' else 'bulk_report.json')).read_text())
    finally:
        (dest/'function_wall.json').write_text(json.dumps(dict(seconds=time.monotonic()-start,hourly=4.5232)))
        vol.commit()
