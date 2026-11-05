"""Independent final ZIP audit against immutable vi48; read-only artifact checks."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import zipfile

from vi41c import digest,indexed,load_zip
from vi48 import fingerprint,write
from vi29 import assert_clean,load_corpus_ids

TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>"))


def audit(out):
    tag=out.name;path=out/f'submission_{tag}.zip'
    with zipfile.ZipFile(path) as z:
        assert len(z.infolist())==1 and z.namelist()==['submission.json']
        assert z.infolist()[0].compress_type==zipfile.ZIP_LZMA and z.testzip() is None
    assert path.stat().st_size<=99_000_000
    ref=load_zip(TEMP/'vi48/submission_vi48.zip')
    ref={r['id']:dict(chunks=digest(r['relevant_chunks']),docs=len(r['relevant_docs'])) for r in ref}
    gc.collect();rows=load_zip(path)
    assert len(rows)==1200 and set(indexed(rows))==set(range(1,1201))
    assert all(digest(r['relevant_chunks'])==ref[r['id']]['chunks'] for r in rows)
    assert all(len(r['relevant_docs'])==ref[r['id']]['docs'] for r in rows)
    assert all(len(r['relevant_docs'])==len(set(r['relevant_docs'])) for r in rows)
    assert all(len(r['relevant_chunks'])==len({(c['doc_id'],c['chunk_text']) for c in r['relevant_chunks']}) for r in rows)
    assert all({c['doc_id'] for c in r['relevant_chunks']}<=set(r['relevant_docs']) for r in rows)
    source=TEMP/'vi45'
    bad=set(map(int,json.loads((source/'bad_ids.json').read_text())))-set(map(int,json.loads((source/'new_ids.json').read_text())))
    clean=assert_clean(rows,bad,load_corpus_ids(source/'links_corpus_vi_en_zh.parquet'))
    report=dict(**fingerprint(path),queries=1200,chunks_equal_vi48=1200,doc_counts_equal_vi48=1200,
        clean=clean,crc=0,entries=1,lzma=True,no_duplicates=True,chunks_within_docs=True)
    write(out/'independent_audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);audit(ap.parse_args().out)
