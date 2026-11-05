"""Read-only b660 verification of received dense-chunk artifacts and build output."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

OUT=Path('D:/r2ai/data/sub/dchunk')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('stage',choices=['input','build'])
    args=ap.parse_args()
    names=['top1000.npz','bulk_report.json','embed_report.json','embedding_complete.json','resume_plan.json']
    result={name:dict(bytes=(OUT/name).stat().st_size,sha256=sha(OUT/name)) for name in names}
    bulk=json.loads((OUT/'bulk_report.json').read_text())
    assert result['top1000.npz']['sha256']==bulk['top1000_sha256']
    if args.stage=='build':
        report=json.loads((OUT/'build_report.json').read_text())
        p=OUT/'submission_dchunk30.zip'
        assert p.stat().st_size==report['zip_bytes'] and sha(p)==report['sha256']
        assert report['queries']==1200 and report['verbatim_bad']==report['validate_exit']==report['check_zip']==0
        with zipfile.ZipFile(p) as z:
            assert z.namelist()==['submission.json'] and z.infolist()[0].compress_type==zipfile.ZIP_LZMA
            sub=json.loads(z.read('submission.json'))
        assert len(sub)==1200 and {q['id'] for q in sub}==set(range(1,1201))
        assert all(len(q['relevant_docs'])==len(q['relevant_chunks'])==30 and
                   all(c['doc_id'] in q['relevant_docs'] for c in q['relevant_chunks']) for q in sub)
        result['build_report']=report
        result['lock_exists']=(OUT.parents[2]/'locks/heavy_job.lock').exists()
        assert not result['lock_exists']
    p=OUT/(args.stage+'_transfer_verify.json')
    p.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
