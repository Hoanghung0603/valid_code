"""Verify and upload each approved bulk file separately; never launches a job."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
from concurrent.futures import ThreadPoolExecutor

BASE = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38')
MODAL = __import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/modalenv/Scripts/modal.exe'
os.environ.update(PYTHONIOENCODING='utf-8', MODAL_PROFILE=__import__("os").environ.get("MODAL_PROFILE", "<MODAL_PROFILE>"))
dest = BASE/'bulk_input'
dest.mkdir(exist_ok=True)
archive = BASE/'bulk_input.zip'
with archive.open('wb') as f:
    for i in range(2):
        f.write((BASE/f'bulk_input.zip.{i}').read_bytes())
with zipfile.ZipFile(archive) as z:
    z.extractall(dest)
manifest = json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
names = manifest['modes']['bulk']
assert len(names) == 658 and manifest['bulk_ready'] == 99675
for name in names:
    data = (dest/name).read_bytes()
    assert len(data) == manifest['files'][name]['bytes']
    assert hashlib.sha256(data).hexdigest() == manifest['files'][name]['sha256']
oldlog = (BASE/'bulk_upload.log').read_text(encoding='utf-8') if (BASE/'bulk_upload.log').exists() else ''
done = {n for n in ['manifest.json', *names] if "'/in/vi38/"+n+"'" in oldlog}
def upload(name):
    subprocess.run([MODAL, 'volume', 'put', 'r2ai-runner', str(dest/name),
                    '/in/vi38/'+name, '--force'], check=True, capture_output=True)
    (dest/(name+'.uploaded')).write_text('ok', encoding='utf-8')
    return name
todo = [n for n in ['manifest.json', *names] if n not in done and not (dest/(n+'.uploaded')).exists()]
print('RESUME UPLOAD', len(done), 'remaining', len(todo), flush=True)
with ThreadPoolExecutor(max_workers=6) as pool:
    for i, name in enumerate(pool.map(upload, todo)):
        if i % 25 == 0:
            print('UPLOADED NEW', i, flush=True)
(BASE/'bulk_upload_verified.json').write_text(json.dumps(dict(files=len(names)+1,
    bytes=sum(manifest['files'][n]['bytes'] for n in names), manifest_sha256=hashlib.sha256(
    (dest/'manifest.json').read_bytes()).hexdigest())), encoding='utf-8')
print('UPLOAD DONE', flush=True)
