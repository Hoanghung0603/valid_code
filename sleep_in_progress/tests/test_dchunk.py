import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dchunk import token_windows, collapse, check_sample


@pytest.mark.parametrize('n,expected', [(0,0),(1,1),(511,1),(512,1),(513,2),(974,2),(975,3),(2000,5)])
def test_full_coverage(n, expected):
    spans = list(token_windows(list(range(n)), [(i,i+1) for i in range(n)]))
    assert len(spans) == expected
    covered = set()
    for j,(s,e,a,b) in enumerate(spans):
        assert 0 < e-s <= 512 and (s,e) == (a,b)
        assert s == j*462
        covered.update(range(s,e))
    assert covered == set(range(n))


def test_unicode_verbatim_offsets_not_decode():
    text = '  bệnh\nViệt\u0301 😀  cuối '
    offsets = [(2,6),(7,12),(13,14),(13,14),(16,20)]
    spans = list(token_windows(list(range(5)), offsets, size=3, overlap=1))
    assert [text[a:b] for _,_,a,b in spans] == [text[2:14], text[13:20]]
    assert spans[0][2:] == (2,14)


@pytest.mark.parametrize('size,overlap',[(0,0),(512,512),(2,-1)])
def test_invalid_window(size,overlap):
    with pytest.raises(ValueError):
        list(token_windows([1],[(0,1)],size,overlap))


def test_doc_max_and_tie():
    meta = {0:dict(doc_id=10,idx=2),1:dict(doc_id=10,idx=2),2:dict(doc_id=20,idx=3)}
    assert collapse([1,2,0],[.8,.8,.7],meta) == [[10,2,.8,1],[20,3,.8,2]]
    assert collapse([1,0,2],[.8,.8,.7],meta)[0][-1] == 0
    with pytest.raises(ValueError):
        collapse([0],[float('nan')],meta)


def test_offsets_validation():
    with pytest.raises(ValueError,match='mismatch'):
        list(token_windows([1,2],[(0,1)]))
    with pytest.raises(ValueError,match='Empty'):
        list(token_windows([1],[(2,2)]))


def test_title_only_in_first_and_exact_whitespace():
    text = 'Tiêu đề\n' + 'nội  dung\n'*600
    import re
    offsets = [m.span() for m in re.finditer(r'\S+',text)]
    windows = list(token_windows(list(range(len(offsets))),offsets))
    assert windows[0][2]==0 and text[windows[0][2]:windows[0][3]].startswith('Tiêu đề\n')
    assert all('Tiêu đề' not in text[a:b] for _,_,a,b in windows[1:])
    assert windows[-1][1]==len(offsets)


def test_real_xlmr_offsets_when_tokenizer_available():
    import os
    directory = os.environ.get('DCHUNK_TOKENIZER')
    if not directory:
        pytest.skip('Set DCHUNK_TOKENIZER to saved bge-m3 tokenizer for offline integration check')
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(directory,local_files_only=True)
    text = 'Tiêu đề bệnh\n'+('  Việt\u0301 Nam 😀 thuốc 500mg\n'*1500)
    e = tok(text,add_special_tokens=False,return_offsets_mapping=True,truncation=False,verbose=False)
    spans = list(token_windows(e['input_ids'],e['offset_mapping']))
    assert len(spans)>10
    assert spans[-1][1]==len(e['input_ids'])
    for s,end,a,b in spans:
        assert text[a:b] and '\ufffd' not in text[a:b]
        seq = [tok.cls_token_id]+e['input_ids'][s:end]+[tok.sep_token_id]
        assert len(seq)<=514 and seq[0]==0 and seq[-1]==2
    assert spans[-1][3]==len(text.rstrip())


def test_sample_detects_corrupt_token_and_offset():
    import numpy as np
    class Tok:
        cls_token_id, sep_token_id = 0, 2
        def __call__(self, text, **kwargs):
            return dict(input_ids=[7,8,9],offset_mapping=[(0,1),(1,2),(2,3)])
    doc = dict(text='abc')
    row = dict(start=0,end=3,tokens=3)
    ids, ptr = np.array([0,7,8,9,2]), np.array([0,5])
    assert check_sample(doc,[(0,row)],ids,ptr,Tok())==3
    with pytest.raises(AssertionError):
        check_sample(doc,[(0,{**row,'end':2})],ids,ptr,Tok())
    with pytest.raises(AssertionError):
        check_sample(doc,[(0,row)],np.array([0,7,99,9,2]),ptr,Tok())


def test_full_count_sample_audit(tmp_path, monkeypatch):
    import json
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import dchunk
    from transformers import AutoTokenizer
    class Tok:
        cls_token_id, sep_token_id = 0, 2
        def __call__(self,text,**kwargs):
            return dict(input_ids=[7]*len(text),offset_mapping=[(i,i+1) for i in range(len(text))])
    monkeypatch.setattr(AutoTokenizer,'from_pretrained',lambda *a,**k:Tok())
    monkeypatch.setattr(dchunk,'OUT',tmp_path)
    monkeypatch.setattr(dchunk,'ROOT',tmp_path)
    (tmp_path/'input').mkdir()
    p = tmp_path/'input/tokens_0000.npz'
    np.savez_compressed(p,ids=np.array([0,7,7,7,2]),ptr=np.array([0,5]),chunk_id=np.array([0]))
    pq.write_table(pa.Table.from_pylist([dict(chunk_id=0,doc_id=10,idx=0,start=0,end=3,tokens=3)]),tmp_path/'chunks.parquet')
    pq.write_table(pa.Table.from_pylist([dict(doc_id=10,idx=0,text='abc')]),tmp_path/'docs.parquet')
    manifest = dict(documents=1,chunks=1,source_tokens=3,embedded_tokens_with_overlap_specials=5,empty_documents=0,
                    files=[dict(name=p.name,chunks=1,tokens=5,bytes=p.stat().st_size,sha256=dchunk.sha(p),
                                metadata='chunks.parquet',documents='docs.parquet')])
    dchunk.write(tmp_path/'export_report.json',manifest)
    dchunk.audit(sample=1)
    rep=json.loads((tmp_path/'audit_report.json').read_text())
    assert rep['sampled_documents']==1 and rep['source_tokens']==3 and rep['sha_errors']==0
    p.write_bytes(b'corrupt')
    with pytest.raises(AssertionError):
        dchunk.audit(sample=1)


def test_budget_prediction_includes_full_overhead():
    import importlib.util
    p=Path(__file__).resolve().parents[1]/'deploy/dchunk_local.py'
    spec=importlib.util.spec_from_file_location('dchunk_local',p)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    p=module.prediction(800_000_000,100_000,10,2.1726)
    assert p['seconds']==10530
    assert p['usd']==pytest.approx(10530*2.1726/3600)
    assert p['usd']>4


def test_selected_flat_preserves_ids_offsets():
    import ast
    import numpy as np
    p=Path(__file__).resolve().parents[1]/'deploy/modal_dchunk.py'
    fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='select_flat')
    ns={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
    ids,ptr,cids=ns['select_flat'](np.array([0,7,2,0,8,9,2,0,5,2]),np.array([0,3,7,10]),np.array([10,11,12]),np.array([10,12]))
    assert ids.tolist()==[0,7,2,0,5,2] and ptr.tolist()==[0,3,6] and cids.tolist()==[10,12]
    ids,ptr,cids=ns['select_flat'](ids,ptr,cids,np.array([99]))
    assert len(ids)==len(cids)==0 and ptr.tolist()==[0]


def test_export_pool_scope(tmp_path,monkeypatch):
    import json
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import dchunk
    monkeypatch.setattr(dchunk,'ROOT',tmp_path)
    monkeypatch.setattr(dchunk,'OUT',tmp_path)
    dchunk.write(tmp_path/'data/sub/vi36/in1/pool_v36.json',{'1':[[10,0,1.0]]})
    dchunk.write(tmp_path/'audit_report.json',dict(alternatives=dict(expanded_pool_documents=1,expanded_pool_tokens=9)))
    pq.write_table(pa.Table.from_pylist([dict(chunk_id=0,doc_id=10,tokens=3),dict(chunk_id=1,doc_id=20,tokens=5),dict(chunk_id=2,doc_id=10,tokens=2)]),tmp_path/'chunks_0000.parquet')
    dchunk.export_pool_scope()
    assert np.load(tmp_path/'pool_scope.npz')['chunk_ids'].tolist()==[0,2]
    rep=json.loads((tmp_path/'pool_scope.json').read_text())
    assert rep['scope']=='expanded_pool' and rep['tokens']==9 and rep['chunks']==2


def test_resume_vectors_and_budget():
    import ast
    import importlib.util
    import numpy as np
    root=Path(__file__).resolve().parents[1]
    p=root/'deploy/modal_dchunk.py'
    fn=next(n for n in ast.parse(p.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='validate_vectors')
    ns={}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(p),'exec'),ns)
    c=np.array([1,2],dtype=np.int64)
    e=np.zeros((2,1024),dtype=np.float16)
    e[:,0]=1
    assert ns['validate_vectors'](e,c,c)['norm_min']==1
    for broken in (e.astype('float32'),e*.5,np.full_like(e,np.nan)):
        with pytest.raises(AssertionError):
            ns['validate_vectors'](broken,c,c)
    with pytest.raises(AssertionError):
        ns['validate_vectors'](e,c,c+1)
    p=root/'deploy/dchunk_local.py'
    spec=importlib.util.spec_from_file_location('local_resume',p)
    m=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    manifest=dict(files=[dict(name='a',tokens=100),dict(name='b',tokens=1000)])
    plan=dict(run_parts=['b'],good=[dict(name='a')])
    b=dict(tokens_per_second=100,loaded_seconds=10,hourly_upper=4.7626)
    assert m.resume_forecast(manifest,plan,b,.6,200)['remaining_tokens']==1000
    with pytest.raises(AssertionError):
        m.resume_forecast(manifest,plan,b,4.9,200)
    with pytest.raises(AssertionError):
        m.resume_forecast(manifest,dict(run_parts=['a'],good=[dict(name='a')]),b,.6,200)


def test_detached_and_server_budget_guards():
    root=Path(__file__).resolve().parents[1]
    local=(root/'deploy/dchunk_local.py').read_text()
    server=(root/'deploy/modal_dchunk.py').read_text()
    assert "'run','--detach'" in local
    assert "'No automatic relaunch'" in local
    assert "prior_upper_usd']+seconds_budget" in server
    assert "partial_bench_min_cosine" in server


@pytest.mark.parametrize('bad_id',[None,1])
def test_build_top30_end_to_end(tmp_path,monkeypatch,bad_id):
    import json
    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq
    import dchunk
    real_script=Path(dchunk.__file__).parent/'evaluate.py'
    run=dchunk.subprocess.run
    def validate_run(args,**kw):
        return run([args[0],str(real_script),*args[2:]],**kw)
    monkeypatch.setattr(dchunk.subprocess,'run',validate_run)
    monkeypatch.setattr(dchunk,'ROOT',tmp_path)
    monkeypatch.setattr(dchunk,'OUT',tmp_path)
    np.savez_compressed(tmp_path/'top1000.npz',qids=np.arange(1,1201),
                        chunk_ids=np.tile(np.arange(1000),(1200,1)),
                        scores=np.tile(np.arange(1000,0,-1,dtype=np.float32)/1000,(1200,1)))
    pq.write_table(pa.Table.from_pylist([dict(chunk_id=c,doc_id=c%250+1,idx=c%250,start=0,end=3,tokens=1) for c in range(1000)]),tmp_path/'chunks_0000.parquet')
    pq.write_table(pa.Table.from_pylist([dict(doc_id=d,idx=d-1,text='abc') for d in range(1,251)]),tmp_path/'docs_0000.parquet')
    corpus=tmp_path/'data/sub/diag19/links_corpus_vi_en_zh.parquet'
    corpus.parent.mkdir(parents=True)
    pq.write_table(pa.table({'id':list(range(1,251))}),corpus)
    for p in ('diag28/bad_ids.json','diag30/new_ids.json'):
        dchunk.write(tmp_path/'data/sub'/p,[])
    dchunk.write(tmp_path/'data/sub/diag28/bad_ids.json',[] if bad_id is None else [bad_id])
    cache={str(q):[[d,d-1,1.0] for d in range(1,251)] for q in range(1,1201)}
    for p in ('vi36/in1/dense_v36.json','vi36/in1/pool_v36.json','vi37/cache_fhyde_v37.json'):
        dchunk.write(tmp_path/'data/sub'/p,cache)
    dchunk.write(tmp_path/'bulk_report.json',dict(scope='full',scope_note='test',documents=200,
                 top1000_sha256=dchunk.sha(tmp_path/'top1000.npz')))
    dchunk.build()
    r=json.loads((tmp_path/'build_report.json').read_text())
    assert r['queries']==1200 and r['validate_exit']==r['check_zip']==r['verbatim_bad']==0
    assert r['zip_entries']==1 and r['zip_compression']=='LZMA'
    assert r['top30_overlap_dense']==r['top30_overlap_fhyde_vi37']==(30 if bad_id is None else 29)
    assert r['excluded_bad_or_outside_pairs']==(0 if bad_id is None else 1200)
    assert r['top200_outside_vi37_pool_total']==0
