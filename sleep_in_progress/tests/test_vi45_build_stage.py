"""Exercise build stage with synthetic cache, scores and fresh text export."""
import hashlib
import json
from pathlib import Path
import sys

import zstandard as zstd

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import vi45
import vi37
import vi44
import dwmax_diag as dw
import rerank_run


def test_build_stage_full_contract(tmp_path,monkeypatch):
    root=tmp_path
    out=root/'data/sub/vi45'
    (out/'modal_input').mkdir(parents=True)
    (out/'scores').mkdir()
    ref,cache,scores,inputs=[],{},[],[]
    for q in range(1,1201):
        text=f'Triệu chứng sốt, đau đầu ở bệnh nhân {q}.'
        c=dict(doc_id=q,chunk_text=text,s=float(q%3),zc=float(q%3),zd=0.)
        c['cid']=vi45.cid(q,c)
        ref.append(dict(id=q,relevant_docs=[q],relevant_chunks=[dict(doc_id=q,chunk_text=text)]))
        cache[str(q)]=[c]
        scores.append(dict(qid=str(q),doc_id=q,cid=c['cid'],logit8c=float(q%5)))
        inputs.append(dict(qid=str(q),doc_id=q,cid=c['cid'],text=text,query='q'))
    for version in (37,44):
        p=root/f'data/sub/vi{version}/submission_vi{version}.json'
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps(ref),encoding='utf-8')
    vi37.writez(out/'candidates_vi45_vi44.json.zst',cache)
    part=out/'modal_input/pairs.part000.jsonl.zst'
    with zstd.open(part,'wt',encoding='utf-8') as f:
        f.write('\n'.join(map(json.dumps,inputs)))
    manifest=dict(scored_candidates=1200,files={part.name:dict(bytes=part.stat().st_size,
        sha256=hashlib.sha256(part.read_bytes()).hexdigest())})
    (out/'prepare_vi44_report.json').write_text(json.dumps(manifest))
    (out/'scores/pairs.part000.logits.json').write_text(json.dumps(scores))
    monkeypatch.setattr(vi37,'ROOT',root)
    monkeypatch.setattr(vi44,'ROOT',root)
    fresh=[]
    def export(wanted,dest,prefix):
        fresh.append((wanted,dest,prefix))
        return {},{},['synthetic_fresh_text']
    monkeypatch.setattr(vi37,'export_texts',export)
    monkeypatch.setattr(rerank_run,'load_docs',lambda files,wanted:{q:ref[q-1]['relevant_chunks'][0]['chunk_text'] for q in wanted})
    def checks(sub,zp,name,dest):
        from evaluate import validate
        assert validate(sub)==[]
        return dict(validate=0,check_zip=0,ids=1200)
    monkeypatch.setattr(vi44,'final_checks',checks)
    vi45.build()
    result=dw.read(out/'build_report_vi45.json')
    offline=dw.read(out/'offline_vi45.json')
    assert result['pairs']==1200 and result['verbatim_bad']==0
    assert result['docs_equal_vi37']==result['chunk_counts_equal_vi44']==1200
    assert offline['jaccard_macro']==1 and offline['replaced_total']==0
    assert fresh[0][0]==set(range(1,1201))
    assert fresh[0][1]==out/'verify_text'
    assert (out/'chunk8b_logits_vi45.json.zst').exists()


def test_preverify_all_candidates_not_only_old_chunks(tmp_path,monkeypatch):
    root=tmp_path
    out=root/'data/sub/vi45'
    out.mkdir(parents=True)
    cache={'1':[dict(doc_id=1,chunk_text='old'),dict(doc_id=2,chunk_text='new') ]}
    vi37.writez(out/'candidates_vi45_vi44.json.zst',cache)
    monkeypatch.setattr(vi37,'ROOT',root)
    def export(wanted,dest,prefix):
        assert wanted=={1,2}
        dest.mkdir()
        p=dest/'part.zst'
        p.write_bytes(b'checked fake export')
        return {},{},[p]
    monkeypatch.setattr(vi37,'export_texts',export)
    monkeypatch.setattr(rerank_run,'load_docs',lambda files,wanted:{1:'old',2:'new'})
    vi45.preverify()
    result=dw.read(out/'preverify_report.json')
    assert result['candidates']==result['docs']==2 and result['verbatim_bad']==0
