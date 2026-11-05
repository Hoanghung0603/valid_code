"""Read-only local vi40 ZIP/SHA/doclist audit against original vi37 ZIP."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from evaluate import validate

temp=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))
directory=temp/'vi40'
zp=directory/'submission_vi40.zip'
report=json.loads((directory/'build_report_vi40.json').read_text(encoding='utf-8'))
h=hashlib.sha256()
with zp.open('rb') as f:
    for block in iter(lambda:f.read(8<<20),b''):
        h.update(block)
assert h.hexdigest()==report['sha256']
assert zp.stat().st_size==report['zip_bytes']<=99_000_000
with zipfile.ZipFile(zp) as z:
    assert z.namelist()==['submission.json']
    assert z.infolist()[0].compress_type==zipfile.ZIP_LZMA
    items=json.loads(z.read('submission.json'))
assert not validate(items)
assert len(items)==1200 and {it['id'] for it in items}==set(range(1,1201))
docs={it['id']:it['relevant_docs'] for it in items}
for it in items:
    seen=set()
    for c in it['relevant_chunks']:
        assert c['doc_id'] in it['relevant_docs']
        pair=(c['doc_id'],c['chunk_text'])
        assert pair not in seen
        seen.add(pair)
with zipfile.ZipFile(temp/'vi37/submission_vi37.zip') as z:
    base=json.loads(z.read('submission.json'))
assert docs=={it['id']:it['relevant_docs'] for it in base}
result=dict(sha256=h.hexdigest(),zip_bytes=zp.stat().st_size,local_validate=0,
    doclists_identical=1200,queries=1200,chunk_count=sum(len(it['relevant_chunks']) for it in items),
    zip_entries=1,compression='LZMA',duplicates=0,chunk_outside_doclist=0,
    verbatim='Full offset validation performed on b660; source text not downloaded locally')
(directory/'local_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
