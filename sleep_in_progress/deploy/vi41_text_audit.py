"""Verify all own exported body text against lengths/candidates, CPU only."""
import hashlib
import json
import zstandard
from vi41_local import TEMP

lengths=json.loads((TEMP/'export_lengths.json').read_text())
cands=json.loads((TEMP/'modal_input/new_cands.json').read_text())
docs={}
for p in sorted((TEMP/'modal_input').glob('docs_new.part*.jsonl.zst')):
    with p.open('rb') as f:
        with zstandard.ZstdDecompressor().stream_reader(f) as stream:
            for line in stream.read().splitlines():
                row=json.loads(line)
                d=str(row['id'])
                assert d not in docs
                assert len(row['text'])==lengths[d]['chars']
                assert row['idx']==lengths[d]['idx']
                docs[d]=row['text']
assert set(docs)==set(lengths)=={str(d) for rs in cands.values() for d,i in rs}
result=dict(documents=len(docs),chars=sum(map(len,docs.values())),empty=sum(not t.strip() for t in docs.values()),
    candidate_pairs=sum(map(len,cands.values())),shape_idx_length_errors=0)
(TEMP/'text_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
