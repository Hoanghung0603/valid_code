"""Bounded vi38-only transfer pack; no crawl or other experiment access."""
import json
from pathlib import Path
import zipfile

base = Path('D:/r2ai/data/sub/vi38')
manifest = json.loads((base/'export/manifest.json').read_text(encoding='utf-8'))
with zipfile.ZipFile(base/'bulk_input.zip', 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as z:
    z.write(base/'export/manifest.json', 'manifest.json')
    for name in manifest['modes']['bulk']:
        z.write(base/'export'/name, name)
print('PACKED', (base/'bulk_input.zip').stat().st_size)
with (base/'bulk_input.zip').open('rb') as f:
    i = 0
    while block := f.read(60000000):
        (base/f'bulk_input.zip.{i}').write_bytes(block)
        i += 1
