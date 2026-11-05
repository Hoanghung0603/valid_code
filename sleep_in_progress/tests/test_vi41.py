import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import vi41


def example():
    return {'subqueries':['đau ngực khi vận động', 'khó thở sau gắng sức',
                          'chẩn đoán bệnh mạch vành', 'điện tâm đồ gắng sức'],
            'hyde':['Đoạn y khoa thứ nhất.', 'Đoạn y khoa thứ hai.', 'Đoạn y khoa thứ ba.']}


def test_parse_strict():
    row = example()
    assert vi41.parse_rewrite(json.dumps(row, ensure_ascii=False)) == row
    assert vi41.parse_rewrite('```json\n'+json.dumps(row)+'\n```') == row


@pytest.mark.parametrize('field,value', [('hyde', []), ('subqueries', ['trùng']*4), ('hyde', ['a','b','c'])])
def test_parse_rejects(field, value):
    row = example()
    row[field] = value
    with pytest.raises(AssertionError):
        vi41.parse_rewrite(json.dumps(row))


def test_rrf_dedup_and_ties():
    rows = vi41.rrf_lists([[[2],[2],[1]], [[1],[2]]])
    assert rows[0][0] == 1
    assert rows[0][1] == pytest.approx(1/61+1/62)


def test_select_outside_clean_per_source():
    selected, why = vi41.select_new({'dense':{'1':[[1],[2],[3],[4]]},
                                   'bm25':{'1':[[3],[5],[6]]}},
                                  {'1':[[1,0,1.]]}, {1,2,3,4,5}, {2}, k=2)
    assert selected == {'1':[3,4,5]}
    assert why['1']['3'] == ['dense','bm25']


def test_fixed_caps():
    assert vi41.K_PER_SOURCE == 20
    assert vi41.MAX_DELTA == 55


def test_fixed_nq_uses_each_reference_count_not_expanded_pool():
    fused={'1':[[9,0,3.],[1,1,2.],[2,2,1.]], '2':[[8,0,3.],[7,1,2.],[6,2,1.]]}
    reference=[{'id':2,'relevant_docs':[7]}, {'id':1,'relevant_docs':[1,2]}]
    before=json.dumps(fused)
    assert vi41.fixed_doclists(fused,reference,{'1':[], '2':[]})=={'1':[9,1], '2':[8]}
    assert json.dumps(fused)==before


def test_fixed_nq_preserves_mandatory_without_growing_count():
    fused={'1':[[9,0,3.],[1,1,2.],[2,2,1.]]}
    ref=[{'id':1,'relevant_docs':[1,2]}]
    assert vi41.fixed_doclists(fused,ref,{'1':[{'doc_id':2}]})=={'1':[9,2]}
    with pytest.raises(AssertionError):
        vi41.fixed_doclists(fused,[],{})
    with pytest.raises(AssertionError):
        vi41.fixed_doclists(fused,[{'id':1,'relevant_docs':[]}],{'1':[{'doc_id':2}]})


def test_fixed_nq_chain_filters_chunks_but_never_changes_fusion():
    import vi29
    pool={'1':[[1,0,3.],[2,1,2.],[3,2,1.]]}
    cands={'1':[[d,d,0,float(d),'text'+str(d)] for d in (1,2,3)]}
    args=(pool,{},pool,{},cands,{'1':10},{'1':[]})
    rr,fh,_,_=vi29.chain_build_v29(*args,dedup_mode='doc_text')
    dl=vi41.fixed_doclists(fh,[{'id':1,'relevant_docs':[99]}],{'1':[]})
    rr2,fh2,sub,_=vi29.chain_build_v29(*args,dedup_mode='doc_text',doclists=dl)
    assert rr2==rr and fh2==fh
    assert sub[0]['relevant_docs']==dl['1'] and len(dl['1'])==1
    assert {c['doc_id'] for c in sub[0]['relevant_chunks']}==set(dl['1'])


def test_fixed_nq_cli_isolates_output_by_default(tmp_path,monkeypatch):
    from contextlib import nullcontext
    import dwmax_diag as dw
    monkeypatch.setattr(vi41,'ROOT',tmp_path)
    monkeypatch.setattr(vi41,'OUT',tmp_path/'data/sub/vi41')
    monkeypatch.setattr(vi41,'NQ_FROM',None)
    monkeypatch.setattr(dw,'owned_lock',lambda *a:nullcontext())
    called=[]
    monkeypatch.setattr(vi41,'stats',lambda:called.append(vi41.OUT))
    monkeypatch.setattr(sys,'argv',['vi41.py','stats','--nq-from','vi37.json'])
    vi41.main()
    assert called==[tmp_path/'data/sub/vi41b']
    monkeypatch.setattr(sys,'argv',['vi41.py','stats','--nq-from','vi37.json','--out',str(tmp_path/'data/sub/vi41')])
    with pytest.raises(AssertionError,match='read-only'):
        vi41.main()


def test_shingle_normalization():
    assert vi41.shingle_hashes('A B C D E F') == vi41.shingle_hashes('a  b c d e f')
    assert len(vi41.shingle_hashes('a b c d e f')) == 2
    assert vi41.shingle_hashes('a b') == set()


def test_repair_only_syntax_outside_strings():
    text='{"a": ["giữ , ] trong chuỗi", ],}'
    assert json.loads(vi41.remove_trailing_commas(text)) == {'a':['giữ , ] trong chuỗi']}


def test_last_rank_fallback_preserves_old_order():
    old={'1':[[9,1,.6],[3,2,.4]]}
    got=vi41.append_last_ranks(old,{'1':[3,7,6]}, {'3':{'idx':2},'7':{'idx':4},'6':{'idx':5}})
    assert got['1'][:2]==old['1']
    assert [r[0] for r in got['1']]==[9,3,7,6]
    assert got['1'][2][2]<.4


def test_merged_inputs_extends_every_query(tmp_path,monkeypatch):
    import copy
    import vi37
    import dwmax_diag as dw
    monkeypatch.setattr(vi41,'OUT',tmp_path)
    monkeypatch.setattr(vi41,'ROOT',tmp_path/'root')
    old={q:[[int(q),int(q),1.]] for q in ('1','2')}
    monkeypatch.setattr(vi37,'inputs',lambda:{k:copy.deepcopy(old) for k in ('pool','q8b','dense','hyde')})
    new={'1':[11],'2':[22]}
    vi41.write(tmp_path/'new_ids.json',new)
    vi41.write(tmp_path/'export_report.json',{'windows':2})
    lengths={'11':{'idx':11,'chars':10,'windows':1},'22':{'idx':22,'chars':10,'windows':1}}
    mapping={str(tmp_path/'modal_out/rr_new.json'):{'1':[[11,11,.5]],'2':[[22,22,.6]]},
        str(tmp_path/'modal_input/new_cands.json'):{'1':[[11,11]],'2':[[22,22]]},
        str(tmp_path/'export_lengths.json'):lengths,
        str(tmp_path/'root/data/sub/vi37/q8b_v37.json'):old,
        str(tmp_path/'root/data/sub/diag37/docwin_max_v37.json'):{},
        str(tmp_path/'modal_out/docwin_new.json.zst'):{'1':{'11':[[0,10,.3]]},'2':{'22':[[0,10,.2]]}}}
    original=dw.read
    monkeypatch.setattr(dw,'read',lambda p:copy.deepcopy(mapping[str(p)]) if str(p) in mapping else original(p))
    data,scores,_=vi41.merged_inputs()
    assert [r[0] for r in data['pool']['1']]==[1,11]
    assert [r[0] for r in data['pool']['2']]==[2,22]
    assert scores=={'1':{'11':.3},'2':{'22':.2}}


def test_appended_rank_fallback_no_rewrite_scores_in_fusion():
    from fuse3 import fuse_q
    old=[[1,0,.9],[2,1,.7]]
    extended=vi41.append_last_ranks({'1':old},{'1':[3]},{'3':{'idx':2}})['1']
    assert extended[:2]==old
    got=fuse_q([[1,0,2.],[2,1,1.],[3,2,0.]],[],extended,(0.,0.,2.),60)
    assert [r[0] for r in got]==[1,2,3]
    assert got[2][2]==round(2/63,8)


def test_input_audit_rejects_cap_and_overlap(tmp_path,monkeypatch):
    import dwmax_diag as dw
    monkeypatch.setattr(vi41,'OUT',tmp_path)
    monkeypatch.setattr(vi41,'ROOT',tmp_path/'root')
    old={str(q):[[q,0,1.]] for q in range(1,1201)}
    monkeypatch.setattr(dw,'read',lambda p:old)
    fresh={str(q):[] for q in range(1,1201)}
    prov={str(q):{} for q in range(1,1201)}
    fresh['1']=[1];prov['1']={'1':['dense']}
    vi41.write(tmp_path/'new_ids.json',fresh)
    vi41.write(tmp_path/'new_provenance.json',prov)
    with pytest.raises(AssertionError):
        vi41.audit_inputs()
    fresh['1']=list(range(10000,10021));prov['1']={str(d):['dense'] for d in fresh['1']}
    vi41.write(tmp_path/'new_ids.json',fresh)
    vi41.write(tmp_path/'new_provenance.json',prov)
    with pytest.raises(AssertionError):
        vi41.audit_inputs()


def test_qselect_only_new_missing_nonempty_pre8b_top(tmp_path,monkeypatch):
    import vi37
    monkeypatch.setattr(vi41,'OUT',tmp_path)
    monkeypatch.setattr(vi41,'merged_inputs',lambda:({'q8b':{'1':[[11,11,1.]]}}, {}, {}))
    monkeypatch.setattr(vi37,'fused_doclist',lambda data,scores:
        ({'1':[[d,d,1.] for d in range(1,60)]},{'1':[1]}))
    vi41.write(tmp_path/'new_ids.json',{'1':[11,12,13,40]})
    vi41.write(tmp_path/'export_lengths.json',{str(d):{'idx':d,'chars':10,'windows':0 if d==13 else 1}
                                             for d in (11,12,13,40)})
    vi41.qselect()
    assert vi41.read(tmp_path/'modal_input/q8b_cands.json')=={'1':[[12,12]]}
    assert vi41.read(tmp_path/'qselect_report.json')['pairs']==1


def test_pre_chunks_same_as_direct_builder(tmp_path,monkeypatch):
    import vi37
    import dwmax_diag as dw
    import rerank_run
    from chunks_from_docwin import build_q_rows,select_q_rows
    monkeypatch.setattr(vi41,'OUT',tmp_path)
    text=' '.join('từ'+str(i) for i in range(1100))
    cands={'1':[[7,0]]}
    wins={'1':{'7':[[0,len(text),1.]]}}
    vi41.write(tmp_path/'modal_input/new_cands.json',cands)
    vi37.writez(tmp_path/'modal_out/docwin_new.json.zst',wins)
    monkeypatch.setattr(rerank_run,'load_docs',lambda *args,**kwargs:{7:text})
    vi41.pre_chunks()
    actual=dw.read(tmp_path/'new_cands_u900.json.zst')
    expected=select_q_rows(build_q_rows(cands['1'],{7:text},wins['1'],900,1000,{}),100000)
    assert actual=={'1':expected}
    assert vi41.read(tmp_path/'pre_chunks_report.json')['verbatim_bad']==0


def test_merge_prechunks_keeps_existing_doc_candidates():
    old={'1':[[7,1,0,1.,'cũ']]}
    prec={'1':[[7,1,0,2.,'mới khác'],[7,1,1,1.,'thêm'],[8,2,0,.5,'doc khác'],[9,3,0,99.,'ngoài final']]}
    merged=vi41.merge_prechunks(old,prec,{'1':[7,8]})
    assert {tuple(r) for r in merged['1']}=={tuple(old['1'][0]),(8,1,0,.5,'doc khác')}


def test_merge_prechunks_equals_direct_final_generation():
    from chunks_from_docwin import build_q_rows,select_q_rows
    from vi24 import merge_chunks
    docs={d:' '.join('từ'+str(i) for i in range(1000)) for d in (7,8,9)}
    wins={str(d):[[0,len(docs[d]),1.]] for d in docs}
    prec={'1':select_q_rows(build_q_rows([[7,0],[8,1],[9,2]],docs,wins,900,1000,{}),100000)}
    dl={'1':[9,8]}
    direct={'1':select_q_rows(build_q_rows([[9,2],[8,1]],docs,wins,900,1000,{}),100000)}
    assert vi41.merge_prechunks({'1':[]},prec,dl)==merge_chunks({'1':[]},direct)
