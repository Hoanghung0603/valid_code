"""Batch tokenizer audit of all exported bulk prompts, before GPU spend."""
import json
from pathlib import Path
from tokenizers import Tokenizer

base = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38')
tok = Tokenizer.from_file(str(base/'tokenizer.json'))
manifest = json.loads((base/'bulk_input/manifest.json').read_text(encoding='utf-8'))
pairs = total = 0
for name in manifest['modes']['bulk']:
    rows = json.loads((base/'bulk_input'/name).read_text(encoding='utf-8'))
    enc = tok.encode_batch([r['prompt'] for r in rows], add_special_tokens=False)
    assert all(len(e.ids) == r['n_tokens'] and len(e.ids) < 4096 for e, r in zip(enc, rows))
    pairs += len(rows)
    total += sum(len(e.ids) for e in enc)
assert pairs == 99675 and total == 78662609
(base/'bulk_token_audit.json').write_text(json.dumps(dict(pairs=pairs, tokens=total, passed=True)), encoding='utf-8')
print('ALL BULK TOKENS PASS', pairs, total, flush=True)
