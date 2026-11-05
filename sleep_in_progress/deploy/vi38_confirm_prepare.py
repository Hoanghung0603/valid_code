"""Verify confirmation input and remaining hard budget before separate-file upload."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

BASE = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38')
MODAL = __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'
RATE = (3.95+4*.0473+48*.008)/3600
os.environ.update(PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
dest = BASE/'confirm_input'
dest.mkdir(exist_ok=True)
with zipfile.ZipFile(BASE/'confirm_input.zip') as z:
    assert all(Path(n).name == n for n in z.namelist())
    z.extractall(dest)
manifest = json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
tokens = pairs = 0
names = manifest['modes']['confirm']
assert names
for name in names:
    data = (dest/name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == manifest['files'][name]['sha256']
    rows = json.loads(data)
    pairs += len(rows)
    tokens += sum(r['n_tokens'] for r in rows)
bulk = json.loads((BASE/'bulk_out/launch_audit.json').read_text(encoding='utf-8'))
spent = bulk['cumulative_upper_usd']
# Measured bulk throughput is more conservative than bench when prefix hit differs.
measured = json.loads((BASE/'bulk_out/report_bulk.json').read_text(encoding='utf-8'))
throughput = min(32262.08, measured['prefill_token_s'])
# Account for per-part Volume commits + tokenization, visible in measured bulk.
overhead_per_part = max(0., (measured['paid_function_wall']-measured['load_wall']-
                            measured['inference_wall'])/len(measured['completed']))
forecast = (tokens/throughput+180+len(names)*overhead_per_part)*RATE*1.30+.12
report = dict(tokens=tokens, pairs=pairs, spent_upper_usd=spent,
              confirm_forecast_upper_usd=forecast, forecast_total_upper_usd=spent+forecast)
(BASE/'confirm_forecast.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
assert spent+forecast <= 8, 'STOP projected cumulative USD >8; no confirm launch'
with (BASE/'confirm_upload.log').open('w', encoding='utf-8') as log:
    for name in ['manifest.json', *names]:
        subprocess.run([MODAL, 'volume', 'put', 'r2ai-runner', str(dest/name),
                        '/in/vi38/'+name, '--force'], check=True, stdout=log, stderr=subprocess.STDOUT)
print('CONFIRM UPLOAD DONE', flush=True)
