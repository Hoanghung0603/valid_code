"""Verify final local zip and gate2 package without exposing private mapping."""
import hashlib
import json
from pathlib import Path
import zipfile

base = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi38')
status = json.loads((base/'final_status.json').read_text(encoding='utf-8'))
zp = base/'submission_vi38b.zip'
assert zp.stat().st_size <= 99000000
assert hashlib.sha256(zp.read_bytes()).hexdigest() == status['submission_vi38b.zip']['sha256']
with zipfile.ZipFile(zp) as z:
    assert z.namelist() == ['submission.json'] and z.testzip() is None
    assert z.infolist()[0].compress_type == zipfile.ZIP_LZMA
    # Do not expand 650MB JSON on local; all semantic checks ran on b660 CPU.
gatezip = base/'gate2_package.zip'
assert hashlib.sha256(gatezip.read_bytes()).hexdigest() == status['gate2_package.zip']['sha256']
dest = base/'judge2'
dest.mkdir(exist_ok=True)
with zipfile.ZipFile(gatezip) as z:
    assert all('..' not in Path(n).parts and not Path(n).is_absolute() for n in z.namelist())
    z.extractall(dest)
blind = json.loads((dest/'gate2_blind.json').read_text(encoding='utf-8'))
assert len(blind) == len({r['code'] for r in blind}) == 300
assert all(set(r) == {'code', 'qid', 'query', 'doc_id', 'text', 'truncated'} for r in blind)
codes = {r['code'] for r in blind}
parts = []
for i in range(6):
    rows = json.loads((dest/f'gate2_s{i+1}.json').read_text(encoding='utf-8'))
    assert len(rows) == 50
    parts += [r['code'] for r in rows]
assert set(parts) == codes and len(parts) == len(set(parts)) == 300
parts = []
for i in range(3):
    rows = json.loads((dest/f'gate2_haiku_s{i+1}.json').read_text(encoding='utf-8'))
    assert len(rows) == 50
    parts += [r['code'] for r in rows]
assert len(parts) == len(set(parts)) == 150 and set(parts) <= codes
public = [p for p in dest.iterdir() if p.is_file()]
verification = {p.name: dict(bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in public}
remote_report_path = base/'gate2_report.json'
if remote_report_path.exists():
    remote_report = json.loads(remote_report_path.read_text(encoding='utf-8'))
    assert verification == remote_report['files']
private = dest/'private/gate2_key.json'
# Private file stays in its own directory; inspect metadata, never print its mapping.
key = json.loads(private.read_text(encoding='utf-8'))
from collections import Counter
assert Counter(r['role'] for r in key.values()) == {'out': 150, 'in': 150}
assert set(key) == codes
assert max(Counter((r['role'], r['qid']) for r in key.values()).values()) <= 2
prior = json.loads((base/'blind.json').read_text(encoding='utf-8'))
assert not codes & {r['code'] for r in prior}
assert not {(r['qid'], r['doc_id']) for r in blind} & {(r['qid'], r['doc_id']) for r in prior}
(base/'gate2_local_verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
print('ZIP SHA and gate2 structure PASS', json.dumps(verification['gate2_blind.json']), flush=True)
