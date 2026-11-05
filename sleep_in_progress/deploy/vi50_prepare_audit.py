"""Read-only independent full U900 enumeration on b660; owns heavy-job lock."""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import time
import zstandard as zstd
from chunking import chunk_document
from dwmax_diag import read,write,owned_lock
from vi48 import fingerprint

ROOT=Path('D:/r2ai')
OUT=ROOT/'data/sub/vi50'


def main():
    with owned_lock(ROOT/'locks/heavy_job.lock',f'vi50 independent audit pid={os.getpid()}'):
        start=time.monotonic()
        prep=read(OUT/'prepare_report.json');assert not prep['top8']
        ctx=read(OUT/'fusion_context.json.zst')
        expected=read(OUT/'expected.json.zst')
        bydoc=defaultdict(list)
        for q,ds in ctx['targets'].items():
            assert ds==[h[0] for h in ctx['rankings'][q][:ctx['counts'][q]+100]]
            for d in ds: bydoc[d].append(q)
        ids={r['cid'] for r in expected};assert len(ids)==len(expected)==539208
        observed=set();docs_seen=set();querydocs=0
        for raw in read(ROOT/'data/sub/vi41b/preverify_report.json')['files']:
            with zstd.open(raw,'rt',encoding='utf-8') as f:
                for line in f:
                    r=json.loads(line);d=r['id']
                    if d not in bydoc:continue
                    assert d not in docs_seen;docs_seen.add(d)
                    chunks=chunk_document(r['text'],'vi',max_units=900)
                    for q in bydoc[d]:
                        querydocs+=1
                        for c in chunks:
                            assert c.text in r['text']
                            identity=hashlib.sha256((q+'\0'+str(d)+'\0'+c.text).encode()).hexdigest()
                            observed.add(identity)
        assert observed==ids and docs_seen==set(bydoc) and querydocs==320960
        files={p.name:fingerprint(p) for p in OUT.glob('*.zst')}
        files.update({p.name:fingerprint(p) for p in (OUT/'modal_input').glob('*.zst')})
        report=dict(all_u900_exact=True,scope_exact=True,query_docs=querydocs,unique_docs=len(docs_seen),pairs=len(ids),
            verbatim_bad=0,files=files,seconds=time.monotonic()-start)
        write(OUT/'independent_prepare_audit.json',report)
        print('AUDIT_DONE '+json.dumps(report),flush=True)


if __name__=='__main__':main()
