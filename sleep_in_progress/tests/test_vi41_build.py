"""Actual validate/check_zip/LZMA/clean/verbatim on 1,200 synthetic pipeline rows."""
from contextlib import nullcontext
import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vi41


@pytest.mark.parametrize('preverified',[False,True])
@pytest.mark.parametrize('fixed_nq',[False,True])
def test_build_real_validators(tmp_path,monkeypatch,preverified,fixed_nq):
    import pyarrow as pa
    import pyarrow.parquet as pq
    import vi29
    import vi37
    import rerank_run
    root=Path(__file__).resolve().parents[1]
    monkeypatch.setattr(vi41,'ROOT',tmp_path)
    monkeypatch.setattr(vi41,'OUT',tmp_path/'out')
    out=vi41.OUT
    # Real validator path; all data and outputs confined to test directory.
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/evaluate.py').write_bytes((root/'scripts/evaluate.py').read_bytes())
    for name,obj in [('sanity_report.json',{'doclist_equal':1200}),
                     ('sanity_chain_report.json',{'chunks_equal':1200}),
                     ('input_audit_report.json',{'source_caps_ok':True})]:
        vi41.write(out/name,obj)
    data={str(q):[[1,0,1.]] for q in range(1,1201)}
    for s in ('pool','q8b','dense','hyde'):
        vi41.write(out/'in1'/(s+'_v41.json'),data)
    vi41.write(out/'modal_out/q8b_new.json',{})
    vi37.writez(out/'cands_v41.json.zst',{})
    ref=[{'id':q,'relevant_chunks':[]} for q in range(1,1201)]
    vi41.write(tmp_path/'data/sub/vi36/in1/ref_chunks_v36.json',ref)
    kp=tmp_path/'ref_k.json';vi41.write(kp,ref)
    monkeypatch.setattr(vi29,'REF_K',kp)
    vi41.write(out/'docwin_max_v41.json',{})
    text='Đoạn y khoa nguyên văn dành riêng kiểm thử.'
    sub=[{'id':q,'relevant_docs':[1],'relevant_chunks':[{'doc_id':1,'chunk_text':text}]} for q in range(1,1201)]
    vi41.write(out/'doclist_v41.json',{str(q):[1] for q in range(1,1201)})
    if fixed_nq:
        reference=tmp_path/'vi37.json'
        vi41.write(reference,[{'id':q,'relevant_docs':[99]} for q in range(1,1201)])
        monkeypatch.setattr(vi41,'NQ_FROM',reference)
        vi41.write(out/'fixed_nq_report.json',{'counts_equal':1200})
        vi41.write(out/'cache_fhyde_v41.json',{})
    monkeypatch.setattr(vi37,'chain_docwin',lambda scores:nullcontext())
    monkeypatch.setattr(vi29,'chain_build_v29',lambda *args,**kwargs:({},{},sub,{'delta':55,'tong_chunk':1200}))
    for p in ('diag28/bad_ids.json','diag30/new_ids.json'):
        vi41.write(tmp_path/'data/sub'/p,[])
    corpus=tmp_path/'data/sub/diag19/links_corpus_vi_en_zh.parquet'
    corpus.parent.mkdir(parents=True)
    pq.write_table(pa.table({'id':[1]}),corpus)
    monkeypatch.setattr(vi37,'export_texts',lambda *args,**kwargs:({}, {}, []))
    monkeypatch.setattr(rerank_run,'load_docs',lambda *args,**kwargs:{1:text})
    if preverified:
        vi41.write(out/'preverify_report.json',{'files':[]})
        vi41.write(out/'verify_lengths.json',{'1':{'chars':len(text)}})
    vi41.build()
    result=vi41.read(out/'build_report.json')
    assert result['queries']==1200 and result['delta']==55
    assert result['check_zip']==result['validate_exit']==result['verbatim_bad']==0
    assert result['assert_clean']['cau']==1200
    assert all(v==0 for k,v in result['assert_clean'].items() if k!='cau')
    name='vi41b' if fixed_nq else 'vi41'
    assert (out/f'submission_{name}.zip').stat().st_size<99000000
