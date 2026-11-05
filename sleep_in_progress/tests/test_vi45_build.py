"""CPU end-to-end packing with 1200 synthetic IDs and exact scored prefix."""
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vi45
from reorder_zip import write_reordered_zip
from evaluate import validate


def test_1200_assemble_pack_validate(tmp_path):
    ref,cache,logits=[],{},{}
    for q in range(1,1201):
        a=dict(doc_id=q,chunk_text=f'Đau đầu và sốt ở bệnh nhân {q}.',zc=0.,zd=0.,s=0.)
        b=dict(doc_id=q,chunk_text=f'Khám và điều trị bệnh nhân {q}.',zc=1.,zd=0.,s=1.)
        for c in (a,b):
            c['cid']=vi45.cid(q,c)
        ref.append(dict(id=q,relevant_docs=[q],relevant_chunks=[dict(doc_id=q,chunk_text=a['chunk_text'])]))
        cache[str(q)]=[a,b]
        logits[str(q)]={b['cid']:0.}
    sub,stats=vi45.assemble(ref,cache,logits)
    assert len(sub)==1200 and sum(s['replaced'] for s in stats)==1200
    zp=tmp_path/'submission_vi45.zip'
    info=write_reordered_zip(sub,zp)
    assert info['zip_bytes']<=99000000
    with zipfile.ZipFile(zp) as z:
        assert z.namelist()==['submission.json'] and z.infolist()[0].compress_type==zipfile.ZIP_LZMA
        assert z.testzip() is None
        packed=json.loads(z.read('submission.json'))
    assert {r['id'] for r in packed}==set(range(1,1201))
    assert validate(packed)==[]
    for r in packed:
        assert r['relevant_docs']==[r['id']]
        assert len(r['relevant_chunks'])==1
        assert r['relevant_chunks'][0]['chunk_text'].startswith('Khám')
