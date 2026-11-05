"""Independent full candidate-verbatim audit using SHA-checked b660 export."""
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from dwmax_diag import read
from rerank_run import load_docs

T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi45')
pre=read(T/'preverify_report.json')
cache=read(T/'candidates_vi45_vi44.json.zst')
files=[]
for name,meta in pre['files'].items():
    p=T/name.replace('\\','/').split('/')[-1]
    assert p.stat().st_size==meta['bytes']
    assert hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
    files.append(str(p))
wanted={int(c['doc_id']) for rows in cache.values() for c in rows}
docs=load_docs(files,wanted=wanted)
assert wanted<=set(docs)
bad=sum(c['chunk_text'] not in docs[int(c['doc_id'])] for rows in cache.values() for c in rows)
assert bad==0
result=dict(docs=len(wanted),chunks=sum(map(len,cache.values())),verbatim_bad=bad,files_sha_bad=0)
(T/'local_candidate_verbatim.json').write_text(json.dumps(result))
print(json.dumps(result))
