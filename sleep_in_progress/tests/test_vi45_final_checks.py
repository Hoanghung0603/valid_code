"""Use actual validate/check_zip/assert_clean on a synthetic 1200-ID archive."""
import json
from pathlib import Path
import sys

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vi42
import vi44
from reorder_zip import write_reordered_zip


def test_real_final_checks(tmp_path,monkeypatch):
    root=tmp_path
    for folder in ('diag28','diag30','diag19'):
        (root/'data/sub'/folder).mkdir(parents=True)
    (root/'data/sub/diag28/bad_ids.json').write_text('[]')
    (root/'data/sub/diag30/new_ids.json').write_text('[]')
    pq.write_table(pa.table({'id':list(range(1,1201)),'url':['https://example.com']*1200}),
        root/'data/sub/diag19/links_corpus_vi_en_zh.parquet')
    sub=[dict(id=q,relevant_docs=[q],relevant_chunks=[dict(doc_id=q,chunk_text=f'Bệnh nhân số {q} cần khám bác sĩ.')])
        for q in range(1,1201)]
    out=root/'result'
    out.mkdir()
    zp=out/'submission_vi45.zip'
    write_reordered_zip(sub,zp)
    monkeypatch.setattr(vi44,'ROOT',root)
    checks=vi44.final_checks(sub,zp,'vi45',out)
    assert checks['validate']==checks['check_zip']==checks['crc']==0
    assert checks['assert_clean']['cau']==1200
    assert checks['compression']=='LZMA'
