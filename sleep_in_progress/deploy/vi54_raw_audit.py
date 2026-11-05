"""Independent local SHA, exact vi51 part order, pool and raw query audit."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from vi54 import fingerprint, iterz, require_raw

TEMP = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))
out = TEMP/'vi54'
prep = json.loads((out/'prepare_report.json').read_text())
old = json.loads((TEMP/'vi51/prepare_report.json').read_text())
seen = set()
parts = {}
for name, meta in prep['parts'].items():
    path = out/'modal_input'/name
    assert fingerprint(path) == {k: meta[k] for k in ('bytes','sha256')}
    rows = require_raw(list(iterz(path)))
    ids = [r['id'] for r in rows]
    original = TEMP/'vi51/modal_input'/name.replace('docs54.', 'docs51.')
    assert fingerprint(original) == old['files'][original.name]
    assert ids == [r['id'] for r in iterz(original)]
    assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == meta['id_order_sha256']
    assert len(ids) == len(set(ids)) == meta['docs'] and not seen.intersection(ids)
    seen.update(ids)
    parts[name] = dict(documents=len(ids),sha256=meta['sha256'],exact_order=True)
for name, meta in prep['files'].items():
    assert fingerprint(out/'modal_input'/name) == meta
queries = require_raw(json.loads((out/'modal_input/queries54.json').read_text(encoding='utf-8')))
pool = json.loads((out/'modal_input/pool.json').read_text())
assert {q['id'] for q in queries} == set(range(1,1201)) and len(queries)==1200
assert seen == {d for ds in pool.values() for d in ds}
assert len(parts)==29 and len(seen)==223065 and sum(map(len,pool.values()))==609662
result = dict(documents=len(seen),pairs=sum(map(len,pool.values())),queries=len(queries),parts=parts,sha_all=True,exact_order=True,nonempty=True)
(out/'raw_audit.json').write_text(json.dumps(result,indent=2))
print('RAW_AUDIT_PASS',len(seen),len(parts),flush=True)
