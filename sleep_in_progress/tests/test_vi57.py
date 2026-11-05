import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import vi57


def c(doc, text, score):
    return {'doc_id': doc, 'chunk_text': text, 'score': score}


def test_selection_and_order():
    row = {'id':1, 'relevant_docs':[1,2], 'relevant_chunks':[{'doc_id':1,'chunk_text':'old'}]}
    candidates = [c(1,'negative',-1), c(1,'zero',0), c(3,'outside',3),
                  c(1,'old',5), c(2,'b',2), c(1,'b',2), c(1,'a',2),
                  c(1,'best',4), c(1,'best',3)]
    selected = vi57.select_row(row,candidates)
    assert [(x['doc_id'],x['chunk_text']) for x in selected] == [(1,'best'),(1,'a'),(1,'b'),(2,'b')]
    final = vi57.make_rows([row], {'1':selected}, 0)[0]
    assert final['relevant_chunks'][0] == row['relevant_chunks'][0]
    assert len(final['relevant_chunks']) == 5


def test_budget_binary_threshold():
    base = [{'id':1,'relevant_docs':[1], 'relevant_chunks':[]}]
    additions = {'1':[c(1,'a',1),c(1,'b',2),c(1,'c',3)]}
    def measure(rows):
        return {'zip_bytes': 10+len(rows[0]['relevant_chunks'])*5}
    rows, threshold, attempts = vi57.fit_budget(base, additions, measure, limit=20)
    assert threshold == 1
    assert [x['chunk_text'] for x in rows[0]['relevant_chunks']] == ['b','c']
    assert attempts[-1]['zip_bytes'] <= 20


def test_budget_no_cut():
    base = [{'id':1,'relevant_docs':[1], 'relevant_chunks':[]}]
    _, t, _ = vi57.fit_budget(base, {'1':[c(1,'x',1)]}, lambda rows:{'zip_bytes':99}, limit=99)
    assert t == 0


def test_extra_vi56_format(tmp_path):
    import json
    import zstandard as zstd
    root = tmp_path
    (root/'in').mkdir()
    (root/'out'/'qwen00').mkdir(parents=True)
    p = root/'in'/'qwen.part000.jsonl.zst'
    rows = [dict(identity='doc',qid='1',doc_id=5,kind='q8b_doc',text='whole'),
            dict(identity='chunk',qid='1',doc_id=5,kind='d8w',text='literal')]
    with zstd.open(p,'wt',encoding='utf8') as f:
        for r in rows: f.write(json.dumps(r)+'\n')
    score = [{k:r[k] for k in ('identity','qid','doc_id','kind')} | {'score':1.0} for r in rows]
    sp = root/'out'/'qwen00'/(p.name+'.scores.json')
    sp.write_text(json.dumps(score),encoding='utf8')
    sp.with_suffix('.complete.json').write_text(json.dumps({'input':vi57.fingerprint(p),'output':vi57.fingerprint(sp),'pairs':2}),encoding='utf8')
    got = list(vi57.source_parts_vi56(root,'extra'))
    assert len(got) == 1 and got[0][0]['cid'] == 'chunk' and got[0][0]['text'] == 'literal'


def test_extra_real_vi56():
    root = vi57.HIST/'modal/<MODAL_PROFILE_D>/r2ai-vi56-data'
    assert sum(1 for _ in vi57.source_parts_vi56(root, 'extra')) == 57513
