"""Independent local ZIP/relation audit; no model/browser/submission."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi41c import load_zip
from vi29 import assert_clean
from vi29 import load_corpus_ids
from vi23 import check_zip
from rerank_run import load_docs

T=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi45')
zp=T/'submission_vi45.zip'
report=json.loads((T/'build_report_vi45.json').read_text())
sha=hashlib.sha256(zp.read_bytes()).hexdigest()
assert sha==report['sha256'] and zp.stat().st_size==report['zip_bytes']<=99000000
with zipfile.ZipFile(zp) as z:
    assert z.namelist()==['submission.json'] and z.infolist()[0].compress_type==zipfile.ZIP_LZMA
    assert z.testzip() is None
    ordered=T/'submission_vi45.json'
    with z.open('submission.json') as src,ordered.open('wb') as dst:
        import shutil
        shutil.copyfileobj(src,dst)
sub=json.loads(ordered.read_text(encoding='utf-8'))
ref=load_zip(T.parent/'vi44/submission_vi44.zip')
by={r['id']:r for r in ref}
assert len(sub)==len(by)==1200 and {r['id'] for r in sub}==set(range(1,1201))
for r in sub:
    assert r['relevant_docs']==by[r['id']]['relevant_docs']
    assert len(r['relevant_chunks'])==len(by[r['id']]['relevant_chunks'])
    keys=[(str(c['doc_id']),c['chunk_text']) for c in r['relevant_chunks']]
    assert len(keys)==len(set(keys))
bad=set(map(int,json.loads((T/'bad_ids.json').read_text())))-set(map(int,json.loads((T/'new_ids.json').read_text())))
corpus=load_corpus_ids(T/'links_corpus_vi_en_zh.parquet')
clean=assert_clean(sub,bad,corpus)
pre=json.loads((T/'preverify_report.json').read_text())
textfiles=[]
for name,meta in pre['files'].items():
    p=T/name.replace('\\','/').split('/')[-1]
    assert p.stat().st_size==meta['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
    textfiles.append(str(p))
wanted={int(c['doc_id']) for r in sub for c in r['relevant_chunks']}
docs=load_docs(textfiles,wanted=wanted)
assert wanted<=set(docs)
verbatim=sum(c['chunk_text'] not in docs[int(c['doc_id'])] for r in sub for c in r['relevant_chunks'])
assert verbatim==0
subprocess.run([sys.executable,str(ROOT/'scripts/evaluate.py'),'validate',str(ordered)],check=True)
assert check_zip(str(ordered),str(zp),str(T/'local_check_zip.json'),'vi45-local',max_zip=99000000)==0
result=dict(sha256=sha,zip_bytes=zp.stat().st_size,ids=1200,docs_equal_vi44=1200,
    chunks_count_equal_vi44=1200,validate=0,check_zip=0,crc=0,structural_clean=clean,
    full_bad_ids_corpus='independent local PASS',verbatim_bad=verbatim,verbatim_chunks=sum(len(r['relevant_chunks']) for r in sub))
(T/'local_final_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
