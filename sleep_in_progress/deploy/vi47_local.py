"""CPU vi47: vi41c doclists, exact vi45 chunks, vi37-size protection rule."""
import gc
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from vi41c import indexed, load_zip, digest, use_chunk_source, merge_row
from reorder_zip import write_reordered_zip
from vi29 import assert_clean, load_corpus_ids
from vi23 import check_zip
from rerank_run import load_docs

T = Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))
O = T/'vi47'
O.mkdir(exist_ok=True)
assert not (O/'submission_vi47.zip').exists(), 'Do not overwrite an existing result'
audit45 = json.loads((T/'vi45/local_final_audit.json').read_text())
assert audit45['verbatim_bad'] == audit45['validate'] == audit45['check_zip'] == 0
source_path = T/'vi45/submission_vi45.zip'
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == audit45['sha256']
ref = {q: {'id':q, 'relevant_docs': r['relevant_docs'],
           'relevant_chunks': [{'doc_id':d} for d in {str(c['doc_id']) for c in r['relevant_chunks']}]}
       for q,r in indexed(load_zip(T/'vi37/submission_vi37.zip')).items()}
candidate = {q: {'id':q, 'relevant_docs':r['relevant_docs']}
             for q,r in indexed(load_zip(T/'vi41c/submission_vi41c.zip')).items()}
assert hashlib.sha256((T/'vi41c/submission_vi41c.zip').read_bytes()).hexdigest() == 'b1b01a9f1f5e93e512273432ca2b3bf75da55951252297a75847806683a85b69'
gc.collect()
source = indexed(load_zip(source_path))
assert set(source) == set(ref) == set(candidate) == set(range(1,1201))
out, per_query, chunk_hashes = [], [], {}
for q, row in source.items():
    assert row['relevant_docs'] == ref[q]['relevant_docs']
    reference = use_chunk_source(ref[q], row, allow_changed_chunk_docs=True)
    merged, missing, short = merge_row(reference, candidate[q])
    assert short == 0 and len(merged['relevant_docs']) == len(ref[q]['relevant_docs'])
    assert merged['relevant_chunks'] == row['relevant_chunks']
    docs = list(map(str, merged['relevant_docs']))
    keys = [(str(c['doc_id']), c['chunk_text']) for c in merged['relevant_chunks']]
    assert len(docs) == len(set(docs)) and len(keys) == len(set(keys))
    assert {d for d,t in keys} <= set(docs)
    old_protected = {str(c['doc_id']) for c in ref[q]['relevant_chunks']}
    new_protected = {d for d,t in keys}
    per_query.append(dict(qid=q, missing_added=missing, removed=missing,
                          new_chunk_bearing_docs=len(new_protected-old_protected),
                          doc_count=len(docs), chunk_bearing_docs=len(new_protected)))
    chunk_hashes[q] = digest(row['relevant_chunks'])
    out.append(merged)
del source, ref, candidate
gc.collect()
zp = O/'submission_vi47.zip'
info = write_reordered_zip(out, zp)
assert info['zip_bytes'] <= 99000000
before = {r['id']:digest(r) for r in out}
bad = set(map(int,json.loads((T/'vi45/bad_ids.json').read_text())))-set(map(int,json.loads((T/'vi45/new_ids.json').read_text())))
clean = assert_clean(out, bad, load_corpus_ids(T/'vi45/links_corpus_vi_en_zh.parquet'))
del out
gc.collect()
jp = O/'submission_vi47.json'
with zipfile.ZipFile(zp) as z:
    assert z.namelist() == ['submission.json'] and z.infolist()[0].compress_type == zipfile.ZIP_LZMA
    assert z.testzip() is None
    with z.open('submission.json') as src, jp.open('wb') as dst:
        shutil.copyfileobj(src,dst)
actual = indexed(json.loads(jp.read_text(encoding='utf-8')))
assert set(actual) == set(range(1,1201))
assert all(digest(r) == before[q] and digest(r['relevant_chunks']) == chunk_hashes[q] for q,r in actual.items())
pre = json.loads((T/'vi45/preverify_report.json').read_text())
textfiles=[]
for name,meta in pre['files'].items():
    p=T/'vi45'/name.replace('\\','/').split('/')[-1]
    assert p.stat().st_size==meta['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==meta['sha256']
    textfiles.append(str(p))
wanted={int(c['doc_id']) for r in actual.values() for c in r['relevant_chunks']}
texts=load_docs(textfiles,wanted=wanted)
assert wanted<=set(texts)
verbatim=sum(c['chunk_text'] not in texts[int(c['doc_id'])] for r in actual.values() for c in r['relevant_chunks'])
assert verbatim==0
del texts
del actual
gc.collect()
subprocess.run([sys.executable,str(ROOT/'scripts/evaluate.py'),'validate',str(jp)],check=True)
assert check_zip(str(jp),str(zp),str(O/'check_zip.json'),'vi47',max_zip=99000000) == 0
report = dict(**info,sha256=hashlib.sha256(zp.read_bytes()).hexdigest(),validate=0,check_zip=0,
    crc=0,ids=1200,compression='LZMA',entries=1,chunks_equal_vi45=1200,doc_counts_equal_vi37=1200,
    duplicate_docs=0,duplicate_chunks=0,chunks_outside_docs=0,assert_clean=clean,
    verbatim_bad=verbatim,verbatim_proof='all chunks tested against SHA-verified fresh vi45 full-text exports',
    source_vi45_sha256=audit45['sha256'],protected_missing_total=sum(r['missing_added'] for r in per_query),
    protected_affected_queries=sum(r['missing_added']>0 for r in per_query),
    new_chunk_bearing_docs_vs_vi37=sum(r['new_chunk_bearing_docs'] for r in per_query),
    chunk_bearing_docs_mean=statistics.mean(r['chunk_bearing_docs'] for r in per_query),per_query=per_query)
(O/'build_report_vi47.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k!='per_query'}),flush=True)
