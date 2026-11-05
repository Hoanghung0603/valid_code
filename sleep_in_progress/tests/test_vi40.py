import sys
from pathlib import Path
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from vi40 import intersection, ranked_windows, union_length, locate, remap, ws_spans


def w(i,a,b):
    return dict(chunk_id=i,start=a,end=b,tokens=b-a)


def test_overlap_tie_and_positive_only():
    rows = [w(3,4,8), w(1,0,4), w(0,10,15)]
    assert [r['chunk_id'] for r in ranked_windows(0,8,rows)] == [1,3]
    assert intersection(0,4,4,8) == 0


@pytest.mark.parametrize('spans,total',[([],0), ([(0,5),(3,9)],9), ([(0,2),(4,6)],4), ([(1,9),(2,4)],8)])
def test_union(spans,total):
    assert union_length(spans) == total


def test_translate_title_body_unicode_offsets():
    body='  bệnh Việt\u0301\nthuốc 😀  '
    full='Tiêu đề\n'+body.rstrip()
    a,b,shift=locate(body,full,'Việt\u0301\nthuốc 😀')
    assert full[a:b]=='Việt\u0301\nthuốc 😀'
    assert a == shift+body.index('Việt')
    with pytest.raises(AssertionError):
        locate(body,full,'missing')


def test_remap_keeps_docs_order_dedup_per_parent():
    items=[dict(id=1,relevant_docs=[20,10],relevant_chunks=[
        dict(doc_id=10,chunk_text='abcdef'),dict(doc_id=10,chunk_text='def'),
        dict(doc_id=20,chunk_text='abcdef')])]
    maps={(10,'abcdef'):[w(0,0,4),w(1,3,6)],(10,'def'):[w(1,3,6)],
          (20,'abcdef'):[w(2,0,4),w(3,3,6)]}
    texts={10:'abcdef',20:'abcdef'}
    one,meta=remap(items,maps,texts,1)
    assert one[0]['relevant_docs']==[20,10]
    assert [(c['doc_id'],c['chunk_text']) for c in one[0]['relevant_chunks']]==[(10,'abcd'),(10,'def'),(20,'abcd')]
    two,_=remap(items,maps,texts,2)
    assert len(two[0]['relevant_chunks'])==4
    assert len(meta[0])==3
    with pytest.raises(ValueError):
        remap(items,maps,texts,3)


def test_docwin_grid_no_redundant_tail():
    assert ws_spans('') == []
    text=' '.join(['x']*551)
    assert len(ws_spans(text))==3
    assert len(text[slice(*ws_spans(text)[0])].split())==300
    assert len(text[slice(*ws_spans(text)[-1])].split())==51


def test_cpu_build_1200_end_to_end(tmp_path, monkeypatch):
    import json
    import zipfile
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zstandard
    import vi40
    import vi29
    root=tmp_path
    dc=root/'data/sub/dchunk'
    v37=root/'data/sub/vi37'
    out=root/'data/sub/vi40'
    dc.mkdir(parents=True)
    (v37/'verify_text').mkdir(parents=True)
    out.mkdir(parents=True)
    body='abc def ghi jkl'
    full='Title\n'+body
    def write(p,obj):
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(obj),encoding='utf-8')
    rows=[dict(id=q,relevant_docs=[10],relevant_chunks=[dict(doc_id=10,chunk_text=body)]) for q in range(1,1201)]
    write(v37/'submission_vi37.json',rows)
    (v37/'verify_text/docs_verify.part000.jsonl.zst').write_bytes(
        zstandard.ZstdCompressor().compress((json.dumps(dict(id=10,text=body))+'\n').encode()))
    pq.write_table(pa.Table.from_pylist([dict(doc_id=10,text=full)]),dc/'docs_0000.parquet')
    pq.write_table(pa.Table.from_pylist([
        dict(doc_id=10,chunk_id=0,start=0,end=13,tokens=5),
        dict(doc_id=10,chunk_id=1,start=10,end=len(full),tokens=4)]),dc/'chunks_0000.parquet')
    write(dc/'export_report.json',dict(content_tokens=512,overlap=50,stride=462,
        files=[dict(documents='docs_0000.parquet',metadata='chunks_0000.parquet')]))
    write(root/'data/sub/diag28/bad_ids.json',[])
    write(root/'data/sub/diag30/new_ids.json',[])
    monkeypatch.setattr(vi40,'ROOT',root)
    monkeypatch.setattr(vi40,'OUT',out)
    monkeypatch.setattr(vi29,'load_corpus_ids',lambda _: {10})
    calls=[]
    monkeypatch.setattr(vi40.subprocess,'run',lambda cmd,check: calls.append(cmd))
    vi40.run()
    assert len(calls)==2 and all('validate' in cmd for cmd in calls)
    rep=json.loads((out/'survey.json').read_text(encoding='utf-8'))
    assert rep['all_union_over_U900']['min']==1
    assert rep['builds']['vi40']['doclists_identical']==1200
    assert rep['builds']['vi40_top2']['chunks']==2400
    for name in ('vi40','vi40_top2'):
        with zipfile.ZipFile(out/f'submission_{name}.zip') as z:
            assert z.namelist()==['submission.json']
            assert z.infolist()[0].compress_type==zipfile.ZIP_LZMA
            got=json.loads(z.read('submission.json'))
        assert len(got)==1200
        assert all(c['chunk_text'] in full for it in got for c in it['relevant_chunks'])
