"""Local full-part provenance audit; never modifies collected outputs."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi51 import iterz
from vi48 import fingerprint,write

TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi51')


def audit(tier):
    prep=json.loads((TEMP/'prepare_report.json').read_text());seen=set();documents=0;pairs=0;seconds=0
    out=TEMP.parent/('vi51'+tier);files=[]
    reports=list((out/'outputs').rglob('bulk_report.json'));assert reports
    query_sha=set()
    for p in reports:
        r=json.loads(p.read_text());seconds+=r['function_seconds'];pairs+=r['pairs']
        assert r['dtype'] in ('fp16','bf16')
        query_sha.add(r['files']['query_embeddings.npz']['sha256'])
        for name,expected in r['files'].items():assert fingerprint(p.parent/name)==expected
        for part in r['parts']:
            name=part['part'];assert name not in seen;seen.add(name)
            rows=list(iterz(TEMP/'modal_input'/name.replace('docs51b.','docs51.')))
            ids=np.array([x['id'] for x in rows]);z=np.load(p.parent/(name+'.emb.npz'))
            assert np.array_equal(z['ids'],ids)
            e=z['emb'];assert np.isfinite(e).all() and np.max(np.abs(np.linalg.norm(e.astype(np.float32),axis=1)-1))<.002
            documents+=len(ids);files.append(dict(part=name,documents=len(ids),**fingerprint(p.parent/(name+'.emb.npz'))))
    assert seen==set(prep['parts']) and documents==prep['documents'] and pairs==prep['pairs']
    assert len(query_sha)==1,'Query vectors must match across calls/workspaces'
    report=dict(parts=len(seen),documents=documents,pairs=pairs,finite=True,exact_ids=True,sha=True,gpu_function_seconds=seconds,files=files)
    write(out/'prebuild_audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':audit(sys.argv[1])
