"""Independent read-only ZIP/CX verification; never builds submission."""
import gc
import json
from pathlib import Path
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from vi55 import read,write,fingerprint,iterz
from vi41c import load_zip,digest
from vi29 import assert_clean,load_corpus_ids
from vi56 import completed_part
OUT=ROOT.parent/'r2ai_deliver/cand/vi59d'


def main():
    report=read(OUT/'build_report.json');zp=OUT/'submission_vi59d.zip'
    assert fingerprint(zp)=={k:report[k] for k in ('bytes','sha256')}
    assert zp.stat().st_size<=99_000_000
    with zipfile.ZipFile(zp) as z:
        assert z.namelist()==['submission.json'] and z.infolist()[0].compress_type==zipfile.ZIP_LZMA
        assert z.testzip() is None
    base=load_zip(OUT/'inputs/vi48/submission_vi48.zip')
    chunks={r['id']:digest(r['relevant_chunks']) for r in base}
    counts={r['id']:len(r['relevant_docs']) for r in base}
    del base;gc.collect()
    rows=load_zip(zp)
    assert len(rows)==1200 and {r['id'] for r in rows}==set(range(1,1201))
    assert all(digest(r['relevant_chunks'])==chunks[r['id']] and len(r['relevant_docs'])==counts[r['id']] for r in rows)
    assert all(len(set(r['relevant_docs']))==len(r['relevant_docs']) for r in rows)
    bad=set(map(int,read(OUT/'inputs/vi45/bad_ids.json')))-set(map(int,read(OUT/'inputs/vi45/new_ids.json')))
    clean=assert_clean(rows,bad,load_corpus_ids(OUT/'inputs/vi45/links_corpus_vi_en_zh.parquet'))
    assert clean==report['assert_clean']
    fresh=read(OUT/'new_ids.json');off=read(OUT/'offline.json')
    newcounts=[len(set(r['relevant_docs'])&set(fresh[str(r['id'])])) for r in rows]
    assert abs(sum(newcounts)-off['new_docs_per_query']['mean']*1200)<1e-8
    assert sum(n==0 for n in newcounts)==off['zero_new_queries']
    cx=OUT/'cx_extra';manifest=read(cx/'MANIFEST.json')['files']
    assert all(fingerprint(cx/p)==m for p,m in manifest.items())
    n=0
    for p in sorted((cx/'out').rglob('*.scores.json')):
        source=cx/'in'/p.name.removesuffix('.scores.json')
        assert completed_part(source,p)
        n+=sum(r['kind']=='d8w' for r in iterz(source))
    assert n==106030
    assert fingerprint(OUT/'cx_extra.zip')==read(OUT/'cx_extra_report.json')
    audit=read(OUT/'input_audit.json')['old_inputs']
    assert all(fingerprint(Path(p))==m for p,m in audit.items())
    assert read(OUT/'build_intent.json')['build_attempts']==1
    result=dict(status='PASS',**fingerprint(zp),ids=1200,chunks_equal_vi48=1200,doc_counts_equal_vi48=1200,
                no_duplicate_docs=True,assert_clean=clean,cx_d8w_pairs=n,build_attempts=1)
    write(OUT/'independent_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
