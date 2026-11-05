"""Check fulltext confirmation token caps and unchanged original document SHA."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from tokenizers import Tokenizer

base = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38')
tok = Tokenizer.from_file(str(base/'tokenizer.json'))
manifest = json.loads((base/'confirm_input/manifest.json').read_text(encoding='utf-8'))
seen, truncated, tokens = set(), 0, 0
for name in manifest['modes']['confirm']:
    raw = (base/'confirm_input'/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['files'][name]['sha256']
    rows = json.loads(raw)
    enc = tok.encode_batch([r['prompt'] for r in rows], add_special_tokens=False)
    for r, e in zip(rows, enc):
        assert r['pair_id'] not in seen and r['role'] == 'A'
        seen.add(r['pair_id'])
        assert r['n_tokens'] == len(e.ids) < 4096
        assert r['text_sha256'] == r['confirm_text_sha256']
        assert hashlib.sha256(r['prompt'].encode('utf-8')).hexdigest() == r['prompt_sha256']
        tokens += len(e.ids)
        truncated += int(r['truncated'])
assert len(seen) == 12500 and truncated == 89 and tokens == 19669679
report = dict(pairs=len(seen), truncated=truncated, tokens=tokens, passed=True)
(base/'confirm_token_audit.json').write_text(json.dumps(report), encoding='utf-8')
print(json.dumps(report), flush=True)
