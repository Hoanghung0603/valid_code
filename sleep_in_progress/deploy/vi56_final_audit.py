"""Independent CPU-only audit of vi56 ZIP; never builds it or changes historical sources."""
import gc
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vi56 import TEMP, read, write, fingerprint
from vi29 import assert_clean, load_corpus_ids
from vi41c import load_zip, digest


def main():
    out=TEMP/'vi56';zp=out/'submission_vi56.zip'
    report=read(out/'build_report.json')
    assert fingerprint(zp)=={k:report[k] for k in ('bytes','sha256')}
    assert zp.stat().st_size<=99_000_000
    with zipfile.ZipFile(zp) as z:
        assert z.namelist()==['submission.json']
        assert z.infolist()[0].compress_type==zipfile.ZIP_LZMA
        assert z.testzip() is None
    base=load_zip(TEMP/'vi48/submission_vi48.zip')
    chunks={r['id']:digest(r['relevant_chunks']) for r in base}
    counts={r['id']:len(r['relevant_docs']) for r in base}
    del base;gc.collect()
    rows=load_zip(zp)
    assert len(rows)==1200 and {r['id'] for r in rows}==set(range(1,1201))
    assert all(digest(r['relevant_chunks'])==chunks[r['id']] for r in rows)
    assert all(len(r['relevant_docs'])==counts[r['id']] for r in rows)
    assert all(len(set(r['relevant_docs']))==len(r['relevant_docs']) for r in rows)
    bad=set(map(int,read(TEMP/'vi45/bad_ids.json')))-set(map(int,read(TEMP/'vi45/new_ids.json')))
    clean=assert_clean(rows,bad,load_corpus_ids(TEMP/'vi45/links_corpus_vi_en_zh.parquet'))
    assert clean==report['assert_clean']
    fresh=read(out/'new_ids.json')
    new_counts=[len(set(r['relevant_docs'])&set(fresh[str(r['id'])])) for r in rows]
    offline=read(out/'offline.json')
    assert abs(sum(new_counts)-offline['new_docs_per_query']['mean']*1200)<1e-8
    assert sum(n==0 for n in new_counts)==offline['zero_new_queries']
    before=read(out/'input_audit.json')['old_inputs']
    assert all(fingerprint(Path(p))==meta for p,meta in before.items())
    result=dict(**fingerprint(zp),ids=1200,chunks_equal_vi48=1200,doc_counts_equal_vi48=1200,
                assert_clean=clean,crc=0,entry='submission.json',compression='LZMA',
                no_duplicate_docs=True,inputs_unchanged=True,build_attempts=read(out/'build_intent.json')['build_attempts'])
    assert result['build_attempts']==1
    write(out/'independent_audit.json',result)
    print('INDEPENDENT_PASS '+json.dumps(result),flush=True)


if __name__=='__main__':main()
