"""Independent local audit of ready bulk part identities, labels, tokens and hashes."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from llm_judge_run import MODEL, REVISION

base = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38')
dest = base/'bulk_input'
manifest = json.loads((dest/'manifest.json').read_text(encoding='utf-8'))
assert manifest['model'] == MODEL and manifest['revision'] == REVISION
counts, queries, pairids, tokens, sizes = Counter(), set(), set(), 0, 0
for name in manifest['modes']['bulk']:
    raw = (dest/name).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['files'][name]['sha256']
    assert len(raw) == manifest['files'][name]['bytes']
    sizes += len(raw)
    for r in json.loads(raw):
        assert r['pair_id'] not in pairids
        pairids.add(r['pair_id'])
        assert r['pair_id'] == f"{r['qid']}:{r['doc_id']}"
        assert r['role'] in ('A', 'B') and r['status'] == 'ready'
        assert r['revision'] == REVISION and r['model'] == MODEL
        assert hashlib.sha256(r['prompt'].encode('utf-8')).hexdigest() == r['prompt_sha256']
        assert r['n_tokens'] < 4096 and len(r['text_sha256']) == 64
        counts[r['role']] += 1
        queries.add(r['qid'])
        tokens += r['n_tokens']
assert len(pairids) == 99675 and tokens == 78662609
report = dict(ready=len(pairids), roles=dict(counts), queries=len(queries), tokens=tokens,
              parts=len(manifest['modes']['bulk']), file_bytes=sizes, abstain=manifest['abstain'])
(base/'bulk_local_input_audit.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report), flush=True)
