import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vi59
import vi56


def test_top200_contains_exact_top100():
    top = {'1':[[d,1-d/1000] for d in range(200)]}
    pool = {'1':[[d,d,1] for d in range(0,200,3)]}
    allowed = set(range(200))-{7,110}
    old = vi56.select(top,pool,allowed)
    new = vi59.select(top,pool,allowed)
    assert old == vi59.select(top,pool,allowed,100)
    assert set(old['1']) <= set(new['1'])
    assert all(d >= 100 for d in set(new['1'])-set(old['1']))
    assert 7 not in new['1'] and 110 not in new['1']


def test_select_rejects_duplicate():
    top = {'1':[[0,1]]*200}
    with pytest.raises(AssertionError): vi59.select(top,{'1':[]},{0})


def test_copy_sha_and_no_overwrite(tmp_path):
    src,dst = tmp_path/'source',tmp_path/'owned/copy'
    src.write_text('immutable',encoding='utf-8')
    m = vi59.copy_checked(src,dst)
    assert m == vi59.copy_checked(src,dst)
    dst.write_text('changed',encoding='utf-8')
    with pytest.raises(AssertionError): vi59.copy_checked(src,dst)


def test_gpu_is_exact_vi56():
    text = (Path(__file__).resolve().parents[1]/'deploy/modal_vi59.py').read_text(encoding='utf-8')
    assert "from vi56 import gpu,write" in text
    assert ".read_only()" in text
    assert "download_weights" not in text
    assert "retries=0" in text
    assert "r2ai-vi59-data" in text


def test_extend_math_is_historical():
    ctx = {'data':{'pool':{'1':[[1,0,2.]]},'q8b':{'1':[[1,0,1.]]},
                   'dense':{'1':[[1,0,.5]]},'hyde':{'1':[[1,0,.2]]}}}
    fresh = {'1':[2]}
    scores = {'1':{'2':dict(bge_doc=3.,q8b_doc=2.,bge_win=4.,d8w=5.,q3e=.8)}}
    data,bge,q3e,d8w = vi56.extend(ctx,{'1':{'1':0.}}, {'1':{'1':.7}}, {'1':{'1':1.}},
                                  fresh,scores,{'2':{'idx':1}})
    assert data['pool']['1'][0] == [2,1,3.]
    assert [r[0] for r in data['dense']['1']] == [1,2]
    assert ctx['data']['pool']['1'] == [[1,0,2.]]
    assert (bge['1']['2'],q3e['1']['2'],d8w['1']['2']) == (4.,.8,5.)


def test_aggregate_keeps_disjoint_extra_and_reuse(tmp_path, monkeypatch):
    vi59.write(tmp_path/'reused_scores.json', {'1':{'1':dict(q3e=.2,bge_doc=1,q8b_doc=2,bge_win=3,d8w=4)}})
    vi59.write(tmp_path/'new_ids.json', {'1':[1,2]})
    monkeypatch.setattr(vi56,'aggregate',lambda path: {'1':{'2':dict(q3e=.3,bge_doc=2,q8b_doc=3,bge_win=4,d8w=5)}})
    result=vi59.aggregate(tmp_path)
    assert list(result['1'])==['1','2']
    assert result['1']['1']['d8w']==4 and result['1']['2']['d8w']==5


def test_load_base_requires_true_vi48(tmp_path):
    with pytest.raises(FileNotFoundError):vi59.load_base(tmp_path)


def test_cx_package_preserves_complete_metadata(tmp_path):
    import zstandard as zstd
    source=tmp_path/'extra/modal_input/qwen.part000.jsonl.zst'
    source.parent.mkdir(parents=True)
    row=dict(identity='unique',qid='1',doc_id=1,idx=0,kind='d8w',query='q',text='nguyên văn')
    source.write_bytes(zstd.ZstdCompressor().compress((json.dumps(row)+'\n').encode()))
    dest=tmp_path/'extra/outputs/qwen00'/(source.name+'.scores.json')
    dest.parent.mkdir(parents=True)
    vi56.process_part(source,dest,lambda rows:[2.5])
    vi59.cx_extra(tmp_path)
    assert (tmp_path/'cx_extra/out/qwen00'/dest.with_suffix('.complete.json').name).exists()
    manifest=vi59.read(tmp_path/'cx_extra/MANIFEST.json')['files']
    assert len(manifest)==3
    assert vi59.read(tmp_path/'cx_extra_report.json')==vi59.fingerprint(tmp_path/'cx_extra.zip')
