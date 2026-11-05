"""vi59: new data Volume, historical weights mounted read-only, unchanged scorer."""
from pathlib import Path
import modal

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
data = modal.Volume.from_name('r2ai-vi59-data',create_if_missing=True)
weights = modal.Volume.from_name('r2ai-vi56-weights').read_only()
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('vllm==0.15.1','zstandard==0.23.0')
         .env({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
         .add_local_dir(str(SCRIPTS),'/root/scripts'))
bge_image = (modal.Image.debian_slim(python_version='3.11')
             .pip_install('torch==2.6.0','transformers==4.51.3','zstandard==0.23.0','numpy==1.26.4','huggingface_hub==0.30.2')
             .env({'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'})
             .add_local_dir(str(SCRIPTS),'/root/scripts'))
app = modal.App('r2ai-vi59',image=image)


@app.function(cpu=2,memory=4096,timeout=300,retries=0,volumes={'/data':data,'/weights':weights})
def ping():
    import sys
    sys.path.insert(0,'/root/scripts')
    from vi56 import fingerprint,read,iterz,write
    from transformers import AutoTokenizer
    data.reload(); weights.reload()
    report = read('/data/prepare_report.json')
    for name,meta in report['parts'].items():
        p = Path('/data/in')/name
        assert fingerprint(p) == {k:meta[k] for k in ('bytes','sha256')}
        ids = [r['identity'] for r in iterz(p)]
        assert len(ids) == meta['count'] == len(set(ids))
    revisions = read('/weights/complete.json')
    assert revisions['qwen']['revision'] == '77d193c791ed757ca307ee72715aa132723da912'
    assert revisions['bge']['revision'] == '953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e'
    for name in ('qwen','bge'): AutoTokenizer.from_pretrained('/weights/'+name,local_files_only=True)
    result = dict(parts=len(report['parts']),weights=revisions,forecast=report['forecast'],weights_readonly=True,
                  gpu_code=fingerprint('/root/scripts/vi56.py'))
    write(Path('/data/ping.json'),result); data.commit()
    return result


def run(stage,job,parts,watchdog):
    import sys,time,uuid,signal,traceback
    sys.path.insert(0,'/root/scripts')
    import vi45
    vi45.MODEL = '/weights/qwen'
    from vi56 import gpu,write
    data.reload(); weights.reload()
    assert Path('/data/ping.json').exists()
    start = time.monotonic(); dest = Path('/data/out')/job
    dest.mkdir(parents=True,exist_ok=True)
    def expired(signum,frame): raise TimeoutError('vi59 watchdog expired')
    previous = signal.signal(signal.SIGALRM,expired); signal.alarm(watchdog)
    try:
        result = gpu(stage,job,parts,watchdog,dest=dest,commit=data.commit)
        result['function_seconds'] = time.monotonic()-start
        write(dest/'report.json',result)
        return result
    except BaseException:
        (dest/'traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
    finally:
        signal.alarm(0); signal.signal(signal.SIGALRM,previous)
        write(dest/('wall.'+uuid.uuid4().hex+'.json'),dict(seconds=time.monotonic()-start)); data.commit()


@app.function(gpu='H100',cpu=4,memory=49152,timeout=2500,retries=0,max_containers=4,
              scaledown_window=2,volumes={'/data':data,'/weights':weights})
def qwen(job,parts,watchdog):
    assert watchdog <= 2400
    return run('qwen',job,parts,watchdog)


@app.function(image=bge_image,gpu='H100',cpu=4,memory=49152,timeout=2500,retries=0,max_containers=2,
              scaledown_window=2,volumes={'/data':data,'/weights':weights})
def bge(job,parts,watchdog):
    assert watchdog <= 2400
    return run('bge',job,parts,watchdog)
