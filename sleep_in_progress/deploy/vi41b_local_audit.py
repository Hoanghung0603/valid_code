"""CPU-only independent verification of downloaded vi41b; never submits."""
import hashlib
import json
from pathlib import Path
import zipfile

TEMP=Path(__import__("os").environ.get("HIST_TMP", "<HIST_TMP>") + '/vi41b')


def main():
    zp=TEMP/'submission_vi41b.zip'
    report=json.loads((TEMP/'build_report.json').read_text(encoding='utf-8'))
    assert report['validate_exit']==report['check_zip']==report['verbatim_bad']==0
    assert report['delta']==55 and report['queries']==1200
    clean=report['assert_clean']
    assert set(clean)=={'cau','bad_doc','bad_chunk','chunk_ngoai_doclist','doc_ngoai_kho','chunk_ngoai_kho','cap_doc_text_trung'}
    assert clean['cau']==1200 and all(v==0 for k,v in clean.items() if k!='cau')
    sha=hashlib.sha256(zp.read_bytes()).hexdigest()
    assert sha==report['sha256'] and zp.stat().st_size==report['zip_bytes']<=99000000
    with zipfile.ZipFile(zp) as z:
        assert z.namelist()==['submission.json'] and z.infolist()[0].compress_type==zipfile.ZIP_LZMA
        assert z.testzip() is None
        rows=json.loads(z.read('submission.json'))
    assert len(rows)==1200 and {r['id'] for r in rows}==set(range(1,1201))
    with zipfile.ZipFile(TEMP.parent/'vi37/submission_vi37.zip') as z:
        old={str(r['id']):r for r in json.loads(z.read(z.namelist()[0]))}
    adds=drops=0
    js=[]
    for row in rows:
        docs=row['relevant_docs']
        before=old[str(row['id'])]['relevant_docs']
        assert len(docs)==len(before)==len(set(docs))
        assert all(c['doc_id'] in docs for c in row['relevant_chunks'])
        assert len(row['relevant_chunks'])==len({(c['doc_id'],c['chunk_text']) for c in row['relevant_chunks']})
        a,b=set(before),set(docs)
        adds+=len(b-a)
        drops+=len(a-b)
        js.append(len(a&b)/len(a|b))
    stats=json.loads((TEMP/'stats_report.json').read_text(encoding='utf-8'))
    assert abs(adds/1200-stats['docs_in_per_query'])<1e-12
    assert abs(drops/1200-stats['docs_out_per_query'])<1e-12
    assert abs(sum(js)/1200-stats['jaccard_mean'])<1e-12
    assert abs(sum(len(r['relevant_chunks']) for r in rows)/1200-stats['chunks_per_query_new'])<1e-12
    offline=json.loads((TEMP/'offline_report.json').read_text(encoding='utf-8'))
    assert abs(sum(len({c['doc_id'] for c in r['relevant_chunks']}) for r in rows)/1200
               -offline['new']['docs_with_chunks_per_query'])<1e-12
    result=dict(zip_bytes=zp.stat().st_size,sha256=sha,sha_matches_b660=True,crc_ok=True,
        single_entry_lzma=True,queries=1200,fixed_counts_equal=1200,structure_bad=0,
        docs_in_per_query=adds/1200,docs_out_per_query=drops/1200,jaccard=sum(js)/1200,cpu_only=True)
    (TEMP/'local_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result))


if __name__=='__main__':
    main()
