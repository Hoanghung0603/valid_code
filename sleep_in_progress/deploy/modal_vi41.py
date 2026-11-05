"""vi41 isolated deployed functions. No retries; no local-entrypoint jobs."""
from pathlib import Path
import modal

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
REVISION = '40c069824f4251a91eefaf281ebe4c544efd3e18'
vol = modal.Volume.from_name('r2ai-runner', create_if_missing=False)
rewrite_image = (modal.Image.debian_slim(python_version='3.11')
    .pip_install('vllm==0.15.1', 'zstandard==0.23.0')
    .env({'HF_HOME': '/hf', 'VI38_MODAL_COMMIT': '1'})
    .run_commands('python -c "from huggingface_hub import snapshot_download; '
                  f"snapshot_download('Qwen/Qwen3-14B', revision='{REVISION}')\"")
    .add_local_file(str(SCRIPTS/'vi41.py'), '/root/vi41.py'))
app = modal.App('r2ai-vi41', image=rewrite_image)


@app.function(gpu='H100', cpu=4, memory=49152, timeout=1100,
              scaledown_window=2, max_containers=1, retries=0, volumes={'/vol': vol})
def rewrite():
    import json
    import subprocess
    import sys
    import time
    start = time.monotonic()
    dest = Path('/vol/out/vi41')
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run([sys.executable, '/root/vi41.py', 'rewrite-gpu',
                        '--root', '/vol'], check=True, timeout=1000)
        return json.loads((dest/'rewrite_report.json').read_text())
    finally:
        (dest/'rewrite_function_wall.json').write_text(json.dumps({
            'seconds': time.monotonic()-start, 'hourly_upper': 4.54+4*.0473+48*.008}))
        vol.commit()
