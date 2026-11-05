"""CPU-only local final zip verification. Never contacts Modal or submits."""
import hashlib
import json
from pathlib import Path
import zipfile
from vi41_local import TEMP,STATE


def main():
    state=json.loads(STATE.read_text())
    upper=sum(r['reserve'] for r in state['calls'].values())
    delta=state['snapshot_delta']
    assert max(upper,delta)<10,(upper,delta)
    zp=TEMP/'submission_vi41.zip'
    if not zp.exists():
        raise FileNotFoundError(zp)
    report=json.loads((TEMP/'build_report.json').read_text())
    assert report['validate_exit']==report['check_zip']==report['verbatim_bad']==0
    assert report['queries']==1200 and report['delta']<=55
    assert json.loads((TEMP/'original_inputs_final_check.json').read_text())['all_unchanged']
    clean=report['assert_clean']
    assert set(clean)=={'cau','bad_doc','bad_chunk','chunk_ngoai_doclist',
                        'doc_ngoai_kho','chunk_ngoai_kho','cap_doc_text_trung'}
    assert clean['cau']==1200
    assert all(v==0 for k,v in clean.items() if k!='cau')
    assert zp.stat().st_size==report['zip_bytes'] and zp.stat().st_size<=99000000
    sha=hashlib.sha256(zp.read_bytes()).hexdigest()
    assert sha==report['sha256'],(sha,report['sha256'])
    with zipfile.ZipFile(zp) as z:
        assert z.namelist()==['submission.json']
        assert z.infolist()[0].compress_type==zipfile.ZIP_LZMA
        assert z.testzip() is None
        rows=json.loads(z.read('submission.json'))
    assert len(rows)==1200 and {r['id'] for r in rows}==set(range(1,1201))
    for row in rows:
        docs=row['relevant_docs']
        assert len(docs)==len(set(docs))
        assert all(c['doc_id'] in docs for c in row['relevant_chunks'])
        assert len(row['relevant_chunks'])==len({(c['doc_id'],c['chunk_text']) for c in row['relevant_chunks']})
    result={'snapshot_delta':delta,'upper':upper,'zip_bytes':zp.stat().st_size,'sha256':sha,
        'sha_matches_b660':True,'queries':1200,'single_entry_lzma':True,'crc_ok':True,'structure_bad':0,
        'cpu_only':True,'billing_snapshot_is_historical':True}
    (TEMP/'local_final_audit.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result))


if __name__=='__main__':
    main()
