import sys
import json
import io
import zipfile
import hashlib
from pathlib import Path
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from p2b_localization import paragraphs, candidate_spans, choose_candidate, verify_group, localize_chunks, digest
from p2b_localization import validate_scores, MODEL, REVISION, CONTROL_SHA256, ROOT


def test_no_invented_boundaries_or_clipped_edges():
    text = 'clipped\n' + '完整中文句子。'*40 + '\n' + 'Medical paragraph. '*40 + '\nclipped'
    units = paragraphs(text)
    assert len(units) == 2
    spans = candidate_spans(text, 50, 500)
    assert spans[0] == (0, len(text))
    assert all(a > 0 and b < len(text) for a,b in spans[1:])
    assert all(a in [s for s,e in units] and b in [e for s,e in units] for a,b in spans[1:])


def test_long_single_paragraph_fallback_and_chinese_without_spaces():
    text = '中文。'*2000
    assert candidate_spans(text, 0, 900) == [(0,len(text))]
    text = 'edge\n'+text+'\nedge'
    assert paragraphs(text) == [(5,len(text)-5)]


@pytest.mark.parametrize('parent,child,expected', [(2.,2.2,0),(2.,2.6,1),(-2.,.5,0)])
def test_selection_margin(parent, child, expected):
    group = pd.DataFrame([dict(candidate_id=0,status='ok',raw_score=parent,span_chars=5000),
                          dict(candidate_id=1,status='ok',raw_score=child,span_chars=1000)])
    assert choose_candidate(group).candidate_id == expected


def test_overflow_is_not_scored_and_ties_fallback():
    group = pd.DataFrame([dict(candidate_id=0,status='too_long',raw_score=float('nan'),span_chars=9000),
                          dict(candidate_id=1,status='ok',raw_score=2.,span_chars=1000),
                          dict(candidate_id=2,status='ok',raw_score=2.,span_chars=2000)])
    assert choose_candidate(group).candidate_id == 0
    assert choose_candidate(group.iloc[:2]).candidate_id == 1


def group_for(text, doc_id):
    start,end = paragraphs(text)[0]
    return pd.DataFrame([
        dict(candidate_id=cid,start=a,end=b,doc_id=doc_id,parent_sha256=digest(text),
             passage_text=text[a:b],text_sha256=digest(text[a:b]),status='ok',
             raw_score=score,span_chars=b-a)
        for cid,a,b,score in [(0,0,len(text),-1.),(1,start,end,3.)]])


def test_source_boundary_and_hash_tampering_rejected():
    text = 'edge\n'+'paragraph '*30+'\nend'
    group = group_for(text, 10)
    verify_group(text, 10, group)
    for field,value in [('start',-1),('doc_id',11),('parent_sha256','bad'),('passage_text','invented')]:
        changed = group.copy()
        changed.loc[1,field] = value
        with pytest.raises(RuntimeError):
            verify_group(text,10,changed)


def test_new_same_doc_duplicates_roll_back_preserving_counts():
    text1 = 'edge\n'+'paragraph '*30+'\nend'
    text2 = 'other edge\n'+'paragraph '*30+'\nother end'
    original = [dict(doc_id=10,chunk_text=text1),dict(doc_id=10,chunk_text=text2)]
    groups = {i:group_for(c['chunk_text'],c['doc_id']) for i,c in enumerate(original)}
    assert localize_chunks(original,groups) == original
    original[1]['doc_id'] = 11
    groups[1] = group_for(text2,11)
    result = localize_chunks(original,groups)
    assert len(result) == 2 and [c['doc_id'] for c in result] == [10,11]
    assert all(c['chunk_text'] != o['chunk_text'] for c,o in zip(result,original))


@pytest.mark.parametrize('tamper', [None, 'revision', 'raw_score', 'input_tokens', 'shard', 'missing'])
def test_score_run_validation_rejects_corruption(tmp_path, tamper):
    frame = pd.DataFrame([dict(pair_id='1:0:0',query_id=1,start=0,end=50),
                          dict(pair_id='2:0:0',query_id=2,start=0,end=50)])
    frame.to_parquet(tmp_path/'p2b_pairs.parquet')
    pair_hash = hashlib.sha256((tmp_path/'p2b_pairs.parquet').read_bytes()).hexdigest()
    code_hash = hashlib.sha256((ROOT/'scripts/p2b_score_worker.py').read_bytes()).hexdigest()
    manifest = dict(schema='p2b_localization_v1',revision=REVISION,control_sha256=CONTROL_SHA256,
                    model=MODEL,max_tokens=1024,policy=dict(min_raw=1.,parent_margin=.5),
                    pair_sha256=pair_hash,pairs=2,queries=2)
    (tmp_path/'p2b_manifest.json').write_text(json.dumps(manifest))
    scores = pd.DataFrame([dict(pair_id=f'{qid}:0:0',revision=REVISION,status='ok',
                                input_tokens=100,raw_score=2.,shard=qid-1) for qid in [1,2]])
    if tamper == 'missing':
        scores = scores.iloc[:1]
    elif tamper:
        scores.loc[0,tamper] = dict(revision='wrong',raw_score=float('nan'),input_tokens=1025,shard=1)[tamper]
    buffer = io.BytesIO()
    scores.to_parquet(buffer)
    data = buffer.getvalue()
    run = dict(input=manifest,score_sha256=hashlib.sha256(data).hexdigest(),truncation=False,
               worker_sha256=code_hash,workers=[dict(shard=i,input_sha256=pair_hash,revision=REVISION,
               model=MODEL,max_tokens=1024,truncation=False,worker_sha256=code_hash) for i in [0,1]])
    target = tmp_path/'scores.zip'
    with zipfile.ZipFile(target,'w') as z:
        z.writestr('scores.parquet',data)
        z.writestr('run.json',json.dumps(run))
    if tamper:
        with pytest.raises(RuntimeError):
            validate_scores(tmp_path,target)
    else:
        _, result = validate_scores(tmp_path,target)
        assert len(result) == 2
