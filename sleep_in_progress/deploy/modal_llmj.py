"""vi38 only. Offline-prepared; NEVER run/deploy without the next budget approval."""
from pathlib import Path
import modal

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
REVISION = '40c069824f4251a91eefaf281ebe4c544efd3e18'
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('vllm==0.15.1', 'zstandard==0.23.0')
         .env({'HF_HOME': '/hf', 'VI38_MODAL_COMMIT': '1'})
         .run_commands('python -c "from huggingface_hub import snapshot_download; '
                       f"snapshot_download('Qwen/Qwen3-14B', revision='{REVISION}')\"")
         .add_local_file(str(SCRIPTS/'llm_judge_run.py'), '/root/llm_judge_run.py'))
vol = modal.Volume.from_name('r2ai-runner', create_if_missing=False)
app = modal.App('r2ai-vi38-llmj', image=image)


@app.function(gpu='H100', cpu=4, memory=49152, timeout=5*3600,
              max_containers=1, retries=0, volumes={'/vol': vol})
def judge(mode: str = 'bench', budget: float = 1., spent: float = 0.):
    import json
    import subprocess
    import sys
    import time
    import uuid
    from pathlib import Path
    start = time.monotonic()
    rate = (3.95 + 4*.0473 + 48*.008)/3600
    dest = Path('/vol/out/vi38')/('bench_out2' if mode == 'bench' else mode+'_out')
    dest.mkdir(parents=True, exist_ok=True)
    # Ledger includes failed/cancelled prior jobs where the container's finally ran.
    prior = sum(json.loads(p.read_text())['usd_est'] for p in dest.glob('ledger_*.json'))
    spent = max(spent, prior)
    seconds = (budget-spent)/rate-60
    if seconds <= 0:
        raise RuntimeError('STOP budget exhausted before startup')
    ledger = dest/f'ledger_{uuid.uuid4().hex}.json'
    try:
        command = [sys.executable, '/root/llm_judge_run.py', '--input', '/vol/in/vi38',
                        '--output', str(dest), '--mode', mode, '--budget', str(budget),
                        '--spent', str(spent)]
        if mode == 'bench':
            command += ['--reference', '/vol/in/vi38/reference146.json']
        subprocess.run(command, check=True, timeout=seconds)
        return json.loads((dest/f'report_{mode}.json').read_text())
    finally:
        wall = time.monotonic()-start
        ledger.write_text(json.dumps(dict(mode=mode, paid_function_wall=wall,
             usd_est=wall*rate, note='Add startup/cancel wall from Modal audit before resume')))
        vol.commit()  # Preserve prior completed hash-parts even if a later part fails.


@app.local_entrypoint()
def main(mode: str = 'bench', budget: float = 1., spent: float = 0.):
    if mode not in ('bench', 'bulk', 'confirm'):
        raise ValueError(mode)
    print(judge.remote(mode, budget, spent))
